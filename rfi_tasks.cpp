#include "rfi_starpu.hpp"

#include <array>
#include <initializer_list>
#include <string>

namespace rfi_starpu {
namespace {
constexpr auto stage_count = static_cast<std::size_t>(Stage::Count);
std::array<starpu_codelet, stage_count> codelets{};
std::array<starpu_perfmodel, stage_count> models{};

KernelArgs kernel_args(const rfi::Config& config) {
  KernelArgs args{};
  args.antennas = config.antennas;
  args.snapshots = config.snapshots;
  args.correlation_lags = config.correlation_lags;
  args.rank = config.subspace_rank;
  args.row_count = config.antennas;
  args.panel_size = config.panel_size;
  args.mgs_passes = config.mgs_passes;
  args.pair_parallel = config.correlation_layout == rfi::CorrelationLayout::PairParallel;
  args.classic = config.gs_kernel == rfi::GsKernelStructure::Classic;
  args.compact = config.compact_gs;
  args.folded = config.correlation_order == rfi::CorrelationOrder::Folded;
  args.column_vectors = config.gs_layout == rfi::GsLayout::ColumnVectors;
  args.has_correlation = config.mode != rfi::Mode::OrthogonalizeOnly;
  args.check_orthogonality = config.mode != rfi::Mode::CorrelationOnly;
  return args;
}

starpu_codelet& codelet(Stage stage) {
  return codelets[static_cast<std::size_t>(stage)];
}

std::uint32_t model_footprint(starpu_task* task) {
  KernelArgs a;
  starpu_codelet_unpack_args(task->cl_arg, &a);
  std::uint32_t hash = 2166136261u;
  // Hash semantic fields, excluding the pass number and struct padding.
  const std::uint64_t fields[] = {a.antennas, a.snapshots, a.correlation_lags,
      a.rank, a.panel_begin, a.panel_end, a.execution_rows, a.row_begin,
      a.row_count, a.panel_size, a.mgs_passes, a.pair_parallel, a.classic,
      a.compact, a.folded, a.column_vectors, a.has_correlation,
      a.check_orthogonality};
  for (auto field : fields) {
    for (unsigned byte = 0; byte < 8; ++byte) {
      hash = (hash ^ static_cast<std::uint8_t>(field)) * 16777619u;
      field >>= 8;
    }
  }
  return hash;
}

template <typename... Bindings>
int submit_tracked(TileBuffers& tile, Stage stage, const KernelArgs& args,
                   Bindings... bindings) {
  auto* task = starpu_task_build(&codelet(stage), bindings...,
                               STARPU_VALUE, &args, sizeof(args), 0);
  if (!task) throw std::runtime_error("create StarPU task");
  track_task(task, tile.tile_id, stage, args);
  return starpu_task_submit(task);
}

void define_codelet(Stage stage, const char* name,
                    std::initializer_list<starpu_data_access_mode> modes) {
  auto& cl = codelet(stage);
  starpu_codelet_init(&cl);
  cl.where = STARPU_CUDA;
  cl.cuda_funcs[0] = cuda_function(stage);
  cl.cuda_flags[0] = STARPU_CUDA_ASYNC;
  cl.nbuffers = modes.size() ? static_cast<int>(modes.size()) : STARPU_VARIABLE_NBUFFERS;
  std::copy(modes.begin(), modes.end(), cl.modes);
  cl.name = name;
  auto& model = models[static_cast<std::size_t>(stage)];
  model.type = STARPU_HISTORY_BASED;
  model.symbol = name;
  model.footprint = model_footprint;
  cl.model = &model;
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
  define_codelet(Stage::Mean, "rfi_mean_v2", {STARPU_R, STARPU_W});
  define_codelet(Stage::Stddev, "rfi_stddev_v2", {STARPU_R, STARPU_R, STARPU_W});
  define_codelet(Stage::Normalize, "rfi_normalize_v2", {STARPU_RW, STARPU_R, STARPU_R});
  define_codelet(Stage::RowCorrelation, "rfi_correlation_row_v2",
                 {STARPU_R, STARPU_R, STARPU_W, STARPU_W});
  define_codelet(Stage::PairCorrelation, "rfi_correlation_pair_v2",
                 {STARPU_R, STARPU_R, STARPU_W, STARPU_W});
  define_codelet(Stage::CorrelationSummary, "rfi_correlation_summary_v2", {STARPU_R, STARPU_W});
  define_codelet(Stage::ReverseNorm, "rfi_reverse_norm_v2", {STARPU_R, STARPU_W});
  define_codelet(Stage::ReverseNormalize, "rfi_reverse_normalize_v2", {STARPU_RW, STARPU_R});
  define_codelet(Stage::ReversePanel, "rfi_reverse_panel_v2", {STARPU_RW});
  define_codelet(Stage::ReverseUpdate, "rfi_reverse_update_v2", {STARPU_RW});
  define_codelet(Stage::FinalSummary, "rfi_final_summary_v2", {STARPU_R, STARPU_R, STARPU_W});
  define_codelet(Stage::Tile, "rfi_tile_v2", {});
}

void validate_starpu_config(const rfi::Config& config) {
  if (config.granularity == "partitioned" && config.partitions > config.antennas) {
    throw std::invalid_argument("--partitions must be <= --antennas");
  }
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
  if (config.granularity == "partitioned" &&
      config.mode != rfi::Mode::OrthogonalizeOnly) {
    starpu_data_filter filter{};
    filter.filter_func = starpu_matrix_filter_vertical_block;
    filter.nchildren = static_cast<unsigned>(config.partitions);
    basis_parts.resize(config.partitions);
    snapshot_parts.resize(config.partitions);
    starpu_data_partition_plan(basis, &filter, basis_parts.data());
    starpu_data_partition_plan(correlation_snapshot, &filter, snapshot_parts.data());
  }
}

TileBuffers::~TileBuffers() {
  if (!basis_parts.empty()) {
    starpu_data_partition_clean(basis, basis_parts.size(), basis_parts.data());
    starpu_data_partition_clean(correlation_snapshot, snapshot_parts.size(), snapshot_parts.data());
  }
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
  check_starpu(submit_tracked(tile, Stage::Mean, args,
      STARPU_R, tile.data,
      STARPU_W, tile.mean), "submit mean");
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
  check_starpu(submit_tracked(tile, Stage::Stddev, args,
      STARPU_R, tile.data,
      STARPU_R, tile.mean,
      STARPU_W, tile.stddev), "submit stddev");
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
  check_starpu(submit_tracked(tile, Stage::Normalize, args,
      STARPU_RW, tile.data,
      STARPU_R, tile.mean,
      STARPU_R, tile.stddev), "submit normalize");
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
  check_starpu(submit_tracked(tile, Stage::RowCorrelation, args,
      STARPU_R, tile.data,
      STARPU_R, tile.lag_offsets,
      STARPU_W, tile.basis,
      STARPU_W, tile.correlation_snapshot), "submit row_correlation");
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
  check_starpu(submit_tracked(tile, Stage::PairCorrelation, args,
      STARPU_R, tile.data,
      STARPU_R, tile.lag_offsets,
      STARPU_W, tile.basis,
      STARPU_W, tile.correlation_snapshot), "submit pair_correlation");
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
  check_starpu(submit_tracked(tile, Stage::CorrelationSummary, args,
      STARPU_R, tile.correlation_snapshot,
      STARPU_W, tile.correlation_rows), "submit correlation_summary");
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
  args.mgs_pass = mgs_pass;
  args.panel_begin = pivot;
  check_starpu(submit_tracked(tile, Stage::ReverseNorm, args,
      STARPU_R, tile.basis,
      STARPU_W, tile.mgs_norm), "submit reverse_norm");
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
  args.mgs_pass = mgs_pass;
  args.panel_begin = pivot;
  check_starpu(submit_tracked(tile, Stage::ReverseNormalize, args,
      STARPU_RW, tile.basis,
      STARPU_R, tile.mgs_norm), "submit reverse_normalize");
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
  args.mgs_pass = mgs_pass;
  args.panel_begin = panel_begin;
  args.panel_end = panel_end;
  check_starpu(submit_tracked(tile, Stage::ReversePanel, args,
      STARPU_RW, tile.basis), "submit reverse_panel");
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
  args.mgs_pass = mgs_pass;
  args.panel_begin = panel_begin;
  args.panel_end = panel_end;
  args.execution_rows = execution_rows;
  check_starpu(submit_tracked(tile, Stage::ReverseUpdate, args,
      STARPU_RW, tile.basis), "submit reverse_update");
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
  check_starpu(submit_tracked(tile, Stage::FinalSummary, args,
      STARPU_R, tile.basis,
      STARPU_R, tile.correlation_rows,
      STARPU_W, tile.summary), "submit final_summary");
}

void submit_partitioned_correlation(TileBuffers& tile, const rfi::Config& config,
                                    rfi::Manifest& manifest) {
  const auto parts = static_cast<unsigned>(tile.basis_parts.size());
  starpu_data_partition_submit(tile.basis, parts, tile.basis_parts.data());
  starpu_data_partition_submit(tile.correlation_snapshot, parts, tile.snapshot_parts.data());
  std::size_t begin = 0;
  for (unsigned part = 0; part < parts; ++part) {
    auto args = kernel_args(config);
    args.row_begin = begin;
    args.row_count = starpu_matrix_get_ny(tile.basis_parts[part]);
    const auto stage = args.pair_parallel ? Stage::PairCorrelation : Stage::RowCorrelation;
    manifest.record(config, tile.tile_id, "correlation_partition", begin,
                    begin + args.row_count, 0,
                    range_2d(args.row_count, config.antennas),
                    float_bytes(config.antennas * config.snapshots),
                    float_bytes(2 * args.row_count * config.antennas));
    check_starpu(submit_tracked(tile, stage, args,
        STARPU_R, tile.data, STARPU_R, tile.lag_offsets,
        STARPU_W, tile.basis_parts[part], STARPU_W, tile.snapshot_parts[part]),
        "submit correlation partition");
    begin += args.row_count;
  }
  // The explicit CPU gather is part of this variant's measured communication.
  starpu_data_unpartition_submit(tile.basis, parts, tile.basis_parts.data(), STARPU_MAIN_RAM);
  starpu_data_unpartition_submit(tile.correlation_snapshot, parts, tile.snapshot_parts.data(), STARPU_MAIN_RAM);
}

void submit_tile(TileBuffers& tile, const rfi::Config& config, rfi::Manifest& manifest) {
  auto args = kernel_args(config);
  manifest.record(config, tile.tile_id, "tile", 0, 0, 0, "whole_tile", 0, 0);
  starpu_data_descr buffers[] = {
      {tile.data, args.has_correlation ? STARPU_RW : STARPU_R},
      {tile.mean, STARPU_W}, {tile.stddev, STARPU_W},
      {tile.basis, args.has_correlation ? STARPU_W : STARPU_RW},
      {tile.correlation_snapshot, STARPU_W}, {tile.lag_offsets, STARPU_R},
      {tile.correlation_rows, args.has_correlation ? STARPU_W : STARPU_R},
      {tile.summary, STARPU_W}, {tile.mgs_norm, STARPU_W}};
  check_starpu(submit_tracked(tile, Stage::Tile, args,
      STARPU_DATA_MODE_ARRAY, buffers, 9), "submit tile");
}

std::size_t expected_tasks(const rfi::Config& config) {
  if (config.granularity == "tile") return rfi::tile_count(config);
  const auto extra = config.granularity == "partitioned" &&
      config.mode != rfi::Mode::OrthogonalizeOnly ? config.partitions - 1 : 0;
  return rfi::tile_count(config) * (rfi::kernels_per_tile(config) + extra);
}

void submit_correlation(TileBuffers &tile,
                        const rfi::Config &config,
                        rfi::Manifest &manifest) {
  if (config.granularity == "partitioned") {
    submit_partitioned_correlation(tile, config, manifest);
  } else if (config.correlation_layout == rfi::CorrelationLayout::PairParallel) {
    submit_pair_correlation(tile, config, manifest);
  } else {
    submit_row_correlation(tile, config, manifest);
  }
}

void submit_batch(std::vector<std::unique_ptr<TileBuffers>> &tiles,
                  const rfi::Config &config, rfi::Manifest &manifest) {
  if (config.granularity == "tile") {
    for (auto& tile : tiles) submit_tile(*tile, config, manifest);
    return;
  }
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
