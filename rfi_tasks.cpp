#include "rfi_starpu.hpp"

#include <array>
#include <atomic>
#include <initializer_list>
#include <string>

namespace rfi_starpu {
namespace {
constexpr auto stage_count = static_cast<std::size_t>(Stage::Count);
std::array<starpu_codelet, stage_count> codelets{};
std::array<std::atomic<std::size_t>, STARPU_NMAXWORKERS> worker_tasks{};

KernelArgs kernel_args(const rfi::Config& config) {
  KernelArgs args{};
  args.antennas = config.antennas;
  args.snapshots = config.snapshots;
  args.correlation_lags = config.correlation_lags;
  args.rank = config.subspace_rank;
  args.folded = config.correlation_order == rfi::CorrelationOrder::Folded;
  args.column_vectors = config.gs_layout == rfi::GsLayout::ColumnVectors;
  args.has_correlation = config.mode != rfi::Mode::OrthogonalizeOnly;
  args.check_orthogonality = config.mode != rfi::Mode::CorrelationOnly;
  return args;
}

starpu_codelet& codelet(Stage stage) {
  return codelets[static_cast<std::size_t>(stage)];
}

void define_codelet(Stage stage, const char* name,
                    std::initializer_list<starpu_data_access_mode> modes) {
  auto& cl = codelet(stage);
  starpu_codelet_init(&cl);
  cl.where = STARPU_CUDA;
  cl.cuda_funcs[0] = cuda_function(stage);
  cl.cuda_flags[0] = STARPU_CUDA_ASYNC;
  cl.nbuffers = static_cast<int>(modes.size());
  std::copy(modes.begin(), modes.end(), cl.modes);
  cl.name = name;
}

template <typename T>
void register_vector(starpu_data_handle_t& handle, std::vector<T>& values,
                     const char* name) {
  starpu_vector_data_register(&handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t>(values.data()),
      static_cast<std::uint32_t>(values.size()), sizeof(T));
  starpu_data_set_name(handle, name);
}

void register_matrix(starpu_data_handle_t& handle, std::vector<float>& values,
                     std::size_t rows, std::size_t cols, const char* name) {
  starpu_matrix_data_register(&handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t>(values.data()),
      static_cast<std::uint32_t>(cols), static_cast<std::uint32_t>(cols),
      static_cast<std::uint32_t>(rows), sizeof(float));
  starpu_data_set_name(handle, name);
}

class HostRead {
public:
  explicit HostRead(starpu_data_handle_t handle) : handle_(handle) {
    check_starpu(starpu_data_acquire(handle_, STARPU_R), "acquire host result");
  }
  ~HostRead() { starpu_data_release(handle_); }
  HostRead(const HostRead&) = delete;
  HostRead& operator=(const HostRead&) = delete;
private:
  starpu_data_handle_t handle_;
};

rfi::work_count float_bytes(rfi::work_count values) {
  return values * sizeof(float);
}
std::string range_1d(std::size_t x) { return std::to_string(x); }
std::string range_2d(std::size_t x, std::size_t y) {
  return std::to_string(x) + "x" + std::to_string(y);
}
} // namespace

void check_starpu(int status, const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) +
                             " failed (StarPU status " + std::to_string(status) + ")");
  }
}

void initialize_codelets() {
  define_codelet(Stage::Mean, "rfi_mean", {STARPU_R, STARPU_W});
  define_codelet(Stage::Stddev, "rfi_stddev", {STARPU_R, STARPU_R, STARPU_W});
  define_codelet(Stage::Normalize, "rfi_normalize", {STARPU_RW, STARPU_R, STARPU_R});
  define_codelet(Stage::RowCorrelation, "rfi_correlation_row",
                 {STARPU_R, STARPU_R, STARPU_W, STARPU_W});
  define_codelet(Stage::PairCorrelation, "rfi_correlation_pair",
                 {STARPU_R, STARPU_R, STARPU_W, STARPU_W});
  define_codelet(Stage::CorrelationSummary, "rfi_correlation_summary", {STARPU_R, STARPU_W});
  define_codelet(Stage::ReverseNorm, "rfi_reverse_norm", {STARPU_R, STARPU_W});
  define_codelet(Stage::ReverseNormalize, "rfi_reverse_normalize", {STARPU_RW, STARPU_R});
  define_codelet(Stage::ReversePanel, "rfi_reverse_panel", {STARPU_RW});
  define_codelet(Stage::ReverseUpdate, "rfi_reverse_update", {STARPU_RW});
  define_codelet(Stage::FinalSummary, "rfi_final_summary", {STARPU_R, STARPU_R, STARPU_W});
}

void record_cuda_task() {
  const int worker = starpu_worker_get_id();
  if (worker >= 0 && worker < STARPU_NMAXWORKERS) {
    worker_tasks[worker].fetch_add(1, std::memory_order_relaxed);
  }
}

void print_worker_stats() {
  std::size_t total = 0, used = 0;
  for (unsigned worker = 0; worker < starpu_worker_get_count(); ++worker) {
    if (starpu_worker_get_type(worker) != STARPU_CUDA_WORKER) continue;
    const auto tasks = worker_tasks[worker].load(std::memory_order_relaxed);
    total += tasks;
    used += tasks != 0;
    std::cout << "WORKER id=" << worker
              << " cuda_device=" << starpu_worker_get_devid(worker)
              << " tasks=" << tasks << '\n';
  }
  std::cout << "EXECUTION completed_tasks=" << total
            << " used_cuda_workers=" << used << '\n';
}

void validate_starpu_config(const rfi::Config& config) {
  if (config.antennas > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("--antennas exceeds StarPU's matrix/vector dimensions");
  }
}

TileBuffers::TileBuffers(std::size_t tile, const rfi::Config& config)
    : tile_id(tile), data_host(rfi::make_samples(tile, config)),
      mean_host(config.antennas, 0.0f), stddev_host(config.antennas, 0.0f),
      basis_host(config.mode == rfi::Mode::OrthogonalizeOnly
                     ? rfi::make_initial_basis(config)
                     : std::vector<float>(config.antennas * config.antennas, 0.0f)),
      correlation_snapshot_host(config.antennas * config.antennas, 0.0f),
      lag_offsets_host(rfi::make_lag_offsets(tile, config)),
      correlation_rows_host(config.antennas), summary_host(1),
      mgs_norm_host(config.subspace_rank, 0.0f) {
  register_matrix(data, data_host, config.antennas, config.snapshots, "data");
  register_vector(mean, mean_host, "mean");
  register_vector(stddev, stddev_host, "stddev");
  register_matrix(basis, basis_host, config.antennas, config.antennas, "basis");
  register_matrix(correlation_snapshot, correlation_snapshot_host,
                  config.antennas, config.antennas, "correlation_snapshot");
  register_vector(lag_offsets, lag_offsets_host, "lag_offsets");
  register_vector(correlation_rows, correlation_rows_host, "correlation_rows");
  register_vector(summary, summary_host, "summary");
  register_vector(mgs_norm, mgs_norm_host, "mgs_norm");
}

TileBuffers::~TileBuffers() {
  // Release handles after read_results acquires the outputs.
  for (auto handle : {data, mean, stddev, basis, correlation_snapshot,
                      lag_offsets, correlation_rows, summary, mgs_norm}) {
    if (handle) starpu_data_unregister_no_coherency(handle);
  }
}

rfi::VerificationResult read_results(
    std::vector<std::unique_ptr<TileBuffers>>& tiles, const rfi::Config& config) {
  rfi::VerificationResult total;
  for (auto& tile : tiles) {
    HostRead summary_read(tile->summary);
    auto current = rfi::verify_summary(config, tile->summary_host[0]);
    if (config.verify_full) {
      HostRead basis_read(tile->basis);
      const auto full = rfi::verify_full_basis(config,
          [&](std::size_t row, std::size_t col) {
            return tile->basis_host[row * config.antennas + col];
          });
      const double scale = std::max(1.0, std::abs(current.basis_checksum));
      current.passed = current.passed && full.passed &&
          std::abs(full.basis_checksum - current.basis_checksum) <= scale * 1.0e-6;
    }
    rfi::merge_verification(total, current);
  }
  return total;
}

// Stage submissions follow below in the same order as the SYCL implementation.
void submit_mean(TileBuffers &tile,
                 const rfi::Config &config, rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const rfi::work_count samples =
      static_cast<rfi::work_count>(antennas) * snapshots;
  manifest.record(config, tile.tile_id, "mean", 0, 0, 0,
                  range_1d(antennas), float_bytes(samples),
                  float_bytes(antennas));
  auto args = kernel_args(config);
  check_starpu(starpu_task_insert(&codelet(Stage::Mean),
      STARPU_R, tile.data,
      STARPU_W, tile.mean,
      STARPU_VALUE, &args, sizeof(args), 0), "submit mean");
}

void submit_stddev(TileBuffers &tile,
                   const rfi::Config &config, rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const rfi::work_count samples =
      static_cast<rfi::work_count>(antennas) * snapshots;
  manifest.record(config, tile.tile_id, "stddev", 0, 0, 0,
                  range_1d(antennas),
                  float_bytes(samples + antennas),
                  float_bytes(antennas));
  auto args = kernel_args(config);
  check_starpu(starpu_task_insert(&codelet(Stage::Stddev),
      STARPU_R, tile.data,
      STARPU_R, tile.mean,
      STARPU_W, tile.stddev,
      STARPU_VALUE, &args, sizeof(args), 0), "submit stddev");
}

void submit_normalize(TileBuffers &tile,
                      const rfi::Config &config, rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const rfi::work_count samples =
      static_cast<rfi::work_count>(antennas) * snapshots;
  manifest.record(config, tile.tile_id, "normalize", 0, 0, 0,
                  range_2d(antennas, snapshots),
                  float_bytes(samples + 2 * antennas),
                  float_bytes(samples));
  auto args = kernel_args(config);
  check_starpu(starpu_task_insert(&codelet(Stage::Normalize),
      STARPU_RW, tile.data,
      STARPU_R, tile.mean,
      STARPU_R, tile.stddev,
      STARPU_VALUE, &args, sizeof(args), 0), "submit normalize");
}

void submit_row_correlation(TileBuffers &tile,
                            const rfi::Config &config,
                            rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const std::size_t correlation_lags = config.correlation_lags;
  const rfi::work_count pairs = rfi::correlation_pair_count(config);
  manifest.record(config, tile.tile_id, rfi::correlation_stage_name(config),
                  0, 0, 0, range_1d(antennas),
                  float_bytes(pairs * correlation_lags * (2 * snapshots)) +
                      pairs * correlation_lags * sizeof(std::uint32_t),
                  float_bytes(static_cast<rfi::work_count>(2) * antennas *
                              antennas));
  auto args = kernel_args(config);
  check_starpu(starpu_task_insert(&codelet(Stage::RowCorrelation),
      STARPU_R, tile.data,
      STARPU_R, tile.lag_offsets,
      STARPU_W, tile.basis,
      STARPU_W, tile.correlation_snapshot,
      STARPU_VALUE, &args, sizeof(args), 0), "submit row_correlation");
}

void submit_pair_correlation(TileBuffers &tile,
                             const rfi::Config &config,
                             rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const std::size_t correlation_lags = config.correlation_lags;
  const rfi::work_count pairs = rfi::correlation_pair_count(config);
  manifest.record(config, tile.tile_id, rfi::correlation_stage_name(config),
                  0, 0, 0, range_2d(antennas, antennas),
                  float_bytes(pairs * correlation_lags * (2 * snapshots)) +
                      pairs * correlation_lags * sizeof(std::uint32_t),
                  float_bytes(static_cast<rfi::work_count>(2) * antennas *
                              antennas));
  auto args = kernel_args(config);
  check_starpu(starpu_task_insert(&codelet(Stage::PairCorrelation),
      STARPU_R, tile.data,
      STARPU_R, tile.lag_offsets,
      STARPU_W, tile.basis,
      STARPU_W, tile.correlation_snapshot,
      STARPU_VALUE, &args, sizeof(args), 0), "submit pair_correlation");
}

void submit_correlation_summary(TileBuffers &tile,
                                const rfi::Config &config,
                                rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  manifest.record(config, tile.tile_id, "correlation_summary", 0, 0, 0,
                  range_1d(antennas),
                  float_bytes(static_cast<rfi::work_count>(antennas) *
                              antennas),
                  static_cast<rfi::work_count>(antennas) *
                      sizeof(rfi::CorrelationRowSummary));
  auto args = kernel_args(config);
  check_starpu(starpu_task_insert(&codelet(Stage::CorrelationSummary),
      STARPU_R, tile.correlation_snapshot,
      STARPU_W, tile.correlation_rows,
      STARPU_VALUE, &args, sizeof(args), 0), "submit correlation_summary");
}

void submit_reverse_norm(TileBuffers &tile,
                         std::size_t pivot, std::size_t mgs_pass,
                         const rfi::Config &config,
                         rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  manifest.record(config, tile.tile_id, "reverse_mgs_norm", pivot,
                  pivot + 1, pivot, range_1d(1), float_bytes(antennas),
                  sizeof(float), mgs_pass);
  auto args = kernel_args(config);
  args.panel_begin = pivot;
  check_starpu(starpu_task_insert(&codelet(Stage::ReverseNorm),
      STARPU_R, tile.basis,
      STARPU_W, tile.mgs_norm,
      STARPU_VALUE, &args, sizeof(args), 0), "submit reverse_norm");
}

void submit_reverse_normalize(TileBuffers &tile,
                              std::size_t pivot, std::size_t mgs_pass,
                              const rfi::Config &config,
                              rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  manifest.record(config, tile.tile_id, "reverse_mgs_normalize", pivot,
                  pivot + 1, pivot, range_1d(antennas),
                  float_bytes(antennas) + sizeof(float),
                  float_bytes(antennas), mgs_pass);
  auto args = kernel_args(config);
  args.panel_begin = pivot;
  check_starpu(starpu_task_insert(&codelet(Stage::ReverseNormalize),
      STARPU_RW, tile.basis,
      STARPU_R, tile.mgs_norm,
      STARPU_VALUE, &args, sizeof(args), 0), "submit reverse_normalize");
}

void submit_reverse_panel(TileBuffers &tile,
                          std::size_t panel_begin, std::size_t panel_end,
                          std::size_t mgs_pass,
                          const rfi::Config &config,
                          rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  constexpr std::size_t local_size = 128;
  const rfi::work_count panel_values =
      static_cast<rfi::work_count>(panel_end - panel_begin) * antennas;
  manifest.record(config, tile.tile_id, "reverse_blocked_mgs_panel",
                  panel_begin, panel_end, panel_begin,
                  range_1d(local_size), float_bytes(panel_values),
                  float_bytes(panel_values), mgs_pass);
  auto args = kernel_args(config);
  args.panel_begin = panel_begin;
  args.panel_end = panel_end;
  check_starpu(starpu_task_insert(&codelet(Stage::ReversePanel),
      STARPU_RW, tile.basis,
      STARPU_VALUE, &args, sizeof(args), 0), "submit reverse_panel");
}

void submit_reverse_update(TileBuffers &tile,
                           std::size_t panel_begin, std::size_t panel_end,
                           std::size_t mgs_pass,
                           const rfi::Config &config,
                           rfi::Manifest &manifest) {
  if (panel_begin == 0) {
    return;
  }
  const std::size_t antennas = config.antennas;
  const std::size_t execution_rows =
      config.compact_gs ? panel_begin : config.subspace_rank;
  const rfi::work_count target_values =
      static_cast<rfi::work_count>(panel_begin) * antennas;
  const rfi::work_count panel_values =
      static_cast<rfi::work_count>(panel_end - panel_begin) * antennas;
  const char *stage =
      config.gs_kernel == rfi::GsKernelStructure::Classic
          ? "reverse_mgs_update"
          : "reverse_blocked_mgs_update";
  manifest.record(config, tile.tile_id, stage,
                  panel_begin, panel_end, panel_begin,
                  range_1d(execution_rows),
                  float_bytes(target_values + panel_values),
                  float_bytes(target_values), mgs_pass);
  auto args = kernel_args(config);
  args.panel_begin = panel_begin;
  args.panel_end = panel_end;
  args.execution_rows = execution_rows;
  check_starpu(starpu_task_insert(&codelet(Stage::ReverseUpdate),
      STARPU_RW, tile.basis,
      STARPU_VALUE, &args, sizeof(args), 0), "submit reverse_update");
}

void submit_final_summary(TileBuffers &tile,
                          const rfi::Config &config,
                          rfi::Manifest &manifest) {
  const std::size_t antennas = config.antennas;
  manifest.record(
      config, tile.tile_id, "final_summary", 0, 0, 0, range_1d(1),
      float_bytes(static_cast<rfi::work_count>(antennas) * antennas) +
          static_cast<rfi::work_count>(antennas) *
              sizeof(rfi::CorrelationRowSummary),
      sizeof(rfi::TileSummary));
  auto args = kernel_args(config);
  check_starpu(starpu_task_insert(&codelet(Stage::FinalSummary),
      STARPU_R, tile.basis,
      STARPU_R, tile.correlation_rows,
      STARPU_W, tile.summary,
      STARPU_VALUE, &args, sizeof(args), 0), "submit final_summary");
}

void submit_correlation(TileBuffers &tile,
                        const rfi::Config &config,
                        rfi::Manifest &manifest) {
  if (config.correlation_layout == rfi::CorrelationLayout::PairParallel) {
    submit_pair_correlation(tile, config, manifest);
  } else {
    submit_row_correlation(tile, config, manifest);
  }
}

void submit_batch(std::vector<std::unique_ptr<TileBuffers>> &tiles,
                  const rfi::Config &config, rfi::Manifest &manifest) {
  if (config.mode != rfi::Mode::OrthogonalizeOnly) {
    for (auto &tile : tiles) {
      submit_mean(*tile, config, manifest);
    }
    for (auto &tile : tiles) {
      submit_stddev(*tile, config, manifest);
    }
    for (auto &tile : tiles) {
      submit_normalize(*tile, config, manifest);
    }
    for (auto &tile : tiles) {
      submit_correlation(*tile, config, manifest);
    }
    for (auto &tile : tiles) {
      submit_correlation_summary(*tile, config, manifest);
    }
  }
  if (config.mode != rfi::Mode::CorrelationOnly) {
    // Stage-major submission makes all independent tiles visible together.
    // Each tile still has an independent true basis dependency chain.
    for (std::size_t mgs_pass = 0; mgs_pass < config.mgs_passes;
         ++mgs_pass) {
      if (config.gs_kernel == rfi::GsKernelStructure::Classic) {
        for (std::size_t pivot_cursor = config.subspace_rank;
             pivot_cursor > 0; --pivot_cursor) {
          const std::size_t pivot = pivot_cursor - 1;
          for (auto &tile : tiles) {
            submit_reverse_norm(*tile, pivot, mgs_pass, config,
                                manifest);
          }
          for (auto &tile : tiles) {
            submit_reverse_normalize(*tile, pivot, mgs_pass, config,
                                     manifest);
          }
          for (auto &tile : tiles) {
            submit_reverse_update(*tile, pivot, pivot + 1, mgs_pass,
                                  config, manifest);
          }
        }
        continue;
      }
      std::size_t panel_end = config.subspace_rank;
      while (panel_end > 0) {
        const std::size_t panel_begin =
            panel_end > config.panel_size ? panel_end - config.panel_size : 0;
        for (auto &tile : tiles) {
          submit_reverse_panel(*tile, panel_begin, panel_end, mgs_pass,
                               config, manifest);
        }
        for (auto &tile : tiles) {
          submit_reverse_update(*tile, panel_begin, panel_end, mgs_pass,
                                config, manifest);
        }
        panel_end = panel_begin;
      }
    }
  }
  for (auto &tile : tiles) {
    submit_final_summary(*tile, config, manifest);
  }
}


} // namespace rfi_starpu
