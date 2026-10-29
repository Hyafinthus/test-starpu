#pragma once

#include "rfi_subspace_extraction_types.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rfi_subspace_extraction {

using work_count = unsigned __int128;

enum class Mode { Full, CorrelationOnly, OrthogonalizeOnly };
enum class CorrelationOrder { Forward, Folded };
enum class CorrelationLayout { RowSerial, PairParallel };
enum class GsMapper { Coarse, Precise };
enum class GsLayout { RowVectors, ColumnVectors };
enum class GsKernelStructure { Blocked, Classic };

struct Config {
  std::size_t antennas = 768;
  std::size_t snapshots = 2048;
  std::size_t bands = 8;
  std::size_t time_blocks = 1;
  std::size_t correlation_lags = 32;
  std::size_t subspace_rank = 512;
  std::size_t panel_size = 1;
  std::size_t mgs_passes = 2;
  std::size_t oversubscribe = 1;
  std::size_t theory_devices = 4;
  Mode mode = Mode::Full;
  CorrelationOrder correlation_order = CorrelationOrder::Forward;
  CorrelationLayout correlation_layout = CorrelationLayout::PairParallel;
  GsMapper gs_mapper = GsMapper::Precise;
  GsLayout gs_layout = GsLayout::RowVectors;
  GsKernelStructure gs_kernel = GsKernelStructure::Blocked;
  double memory_limit_gib = 0.0;
  bool compact_gs = false;
  bool verify = true;
  bool verify_full = false;
  bool print_manifest = false;
};

inline const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Full:
    return "full";
  case Mode::CorrelationOnly:
    return "correlation";
  case Mode::OrthogonalizeOnly:
    return "orthogonalize";
  }
  return "unknown";
}

inline const char *correlation_order_name(CorrelationOrder order) {
  return order == CorrelationOrder::Forward ? "forward" : "folded";
}

inline const char *correlation_layout_name(CorrelationLayout layout) {
  return layout == CorrelationLayout::RowSerial ? "row" : "pair";
}

inline const char *gs_mapper_name(GsMapper mapper) {
  return mapper == GsMapper::Coarse ? "coarse" : "precise";
}

inline const char *gs_layout_name(GsLayout layout) {
  return layout == GsLayout::RowVectors ? "row" : "column";
}

inline const char *gs_kernel_name(GsKernelStructure kernel) {
  return kernel == GsKernelStructure::Blocked ? "blocked" : "classic";
}

inline const char *correlation_stage_name(const Config &config) {
  return config.correlation_lags > 1 ? "lagged_correlation_energy"
                                     : "correlation";
}

inline std::string work_count_string(work_count value) {
  if (value == 0) {
    return "0";
  }
  std::string text;
  while (value != 0) {
    text.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(text.begin(), text.end());
  return text;
}

inline bool checked_add(std::size_t lhs, std::size_t rhs,
                        std::size_t &result) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

inline bool checked_mul(std::size_t lhs, std::size_t rhs,
                        std::size_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

inline bool parse_size(const std::string &text, std::size_t &value) {
  if (text.empty() || text.front() == '-') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

inline bool parse_bool(const std::string &text, bool &value) {
  if (text == "1" || text == "true" || text == "TRUE") {
    value = true;
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE") {
    value = false;
    return true;
  }
  return false;
}

inline bool parse_double(const std::string &text, double &value) {
  if (text.empty()) {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      !std::isfinite(parsed)) {
    return false;
  }
  value = parsed;
  return true;
}

inline void print_help(const char *program, bool celerity_options,
                       std::size_t default_oversubscribe = 1) {
  std::cout
      << "RFI subspace-extraction miniapp\n"
      << "Usage: " << program << " [options]\n"
      << "  --antennas <int>             Array channels/features (default 768)\n"
      << "  --snapshots <int>            Samples per frequency band (default 2048)\n"
      << "  --bands <int>                Independent frequency bands (default 8)\n"
      << "  --time-blocks <int>          Independent observation blocks (default 1)\n"
      << "  --correlation-lags <int>     Circular lags per pair (default 32)\n"
      << "  --rank <int>                 Extracted subspace rows (default 512)\n"
      << "  --panel <int>                Reverse blocked-MGS panel (default 1)\n"
      << "  --mgs-passes <int>           Reorthogonalization passes (default 2)\n"
      << "  --gs-layout <row|column>     Basis-vector orientation (default row)\n"
      << "  --gs-kernel <blocked|classic> Fused panel or 3-stage MGS"
         " (default blocked)\n"
      << "  --mode <full|correlation|orthogonalize>\n"
      << "  --covariance-order <forward|folded>\n"
      << "  --correlation-layout <pair|row>\n"
      << "  --compact-gs <0|1>           Submit only the active MGS prefix\n"
      << "  --memory-limit-gib <real>    Fail above logical state; 0 is unlimited\n"
      << "  --theory-devices <int>       Coarse-chunk report width (default 4)\n"
      << "  --verify <0|1>               Validate device summaries (default 1)\n"
      << "  --verify-full <0|1>          Materialize and validate full bases\n"
      << "  --print-manifest <0|1>       Print every submitted kernel\n";
  if (celerity_options) {
    std::cout
        << "  --oversubscribe <int>        Celerity chunks/device hint (default "
        << default_oversubscribe << "; 1 disables extra subdivision)\n"
        << "  --gs-mapper <coarse|precise> Celerity MGS mapper (default precise)\n";
  }
  std::cout << "  --help                        Show this message\n";
}

inline std::size_t tile_count(const Config &config);
inline std::size_t logical_bytes_per_tile(const Config &config);

inline Config parse_config(int argc, char **argv, bool celerity_options,
                           std::size_t default_oversubscribe = 1) {
  Config config;
  config.oversubscribe = default_oversubscribe;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help") {
      print_help(argv[0], celerity_options, default_oversubscribe);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const std::string value = argv[++index];
    bool parsed = false;
    if (option == "--antennas") {
      parsed = parse_size(value, config.antennas);
    } else if (option == "--snapshots") {
      parsed = parse_size(value, config.snapshots);
    } else if (option == "--bands") {
      parsed = parse_size(value, config.bands);
    } else if (option == "--time-blocks") {
      parsed = parse_size(value, config.time_blocks);
    } else if (option == "--correlation-lags") {
      parsed = parse_size(value, config.correlation_lags);
    } else if (option == "--rank") {
      parsed = parse_size(value, config.subspace_rank);
    } else if (option == "--panel") {
      parsed = parse_size(value, config.panel_size);
    } else if (option == "--mgs-passes") {
      parsed = parse_size(value, config.mgs_passes);
    } else if (option == "--gs-layout") {
      if (value == "row") {
        config.gs_layout = GsLayout::RowVectors;
        parsed = true;
      } else if (value == "column") {
        config.gs_layout = GsLayout::ColumnVectors;
        parsed = true;
      }
    } else if (option == "--gs-kernel") {
      if (value == "blocked") {
        config.gs_kernel = GsKernelStructure::Blocked;
        parsed = true;
      } else if (value == "classic") {
        config.gs_kernel = GsKernelStructure::Classic;
        parsed = true;
      }
    } else if (option == "--theory-devices") {
      parsed = parse_size(value, config.theory_devices);
    } else if (option == "--memory-limit-gib") {
      parsed = parse_double(value, config.memory_limit_gib) &&
               config.memory_limit_gib >= 0.0;
    } else if (option == "--compact-gs") {
      parsed = parse_bool(value, config.compact_gs);
    } else if (option == "--verify") {
      parsed = parse_bool(value, config.verify);
    } else if (option == "--verify-full") {
      parsed = parse_bool(value, config.verify_full);
    } else if (option == "--print-manifest") {
      parsed = parse_bool(value, config.print_manifest);
    } else if (option == "--oversubscribe" && celerity_options) {
      parsed = parse_size(value, config.oversubscribe);
    } else if (option == "--gs-mapper" && celerity_options) {
      if (value == "coarse") {
        config.gs_mapper = GsMapper::Coarse;
        parsed = true;
      } else if (value == "precise") {
        config.gs_mapper = GsMapper::Precise;
        parsed = true;
      }
    } else if (option == "--mode") {
      if (value == "full") {
        config.mode = Mode::Full;
        parsed = true;
      } else if (value == "correlation") {
        config.mode = Mode::CorrelationOnly;
        parsed = true;
      } else if (value == "orthogonalize") {
        config.mode = Mode::OrthogonalizeOnly;
        parsed = true;
      }
    } else if (option == "--covariance-order") {
      if (value == "forward") {
        config.correlation_order = CorrelationOrder::Forward;
        parsed = true;
      } else if (value == "folded") {
        config.correlation_order = CorrelationOrder::Folded;
        parsed = true;
      }
    } else if (option == "--correlation-layout") {
      if (value == "row") {
        config.correlation_layout = CorrelationLayout::RowSerial;
        parsed = true;
      } else if (value == "pair") {
        config.correlation_layout = CorrelationLayout::PairParallel;
        parsed = true;
      }
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
    if (!parsed) {
      throw std::invalid_argument("invalid value for " + option + ": " +
                                  value);
    }
  }

  if (config.antennas == 0 || config.snapshots == 0 || config.bands == 0 ||
      config.time_blocks == 0 || config.correlation_lags == 0 ||
      config.subspace_rank == 0 || config.panel_size == 0 ||
      config.mgs_passes == 0 || config.oversubscribe == 0 ||
      config.theory_devices == 0) {
    throw std::invalid_argument("all integer parameters must be positive");
  }
  if (config.subspace_rank > config.antennas) {
    throw std::invalid_argument("--rank must not exceed --antennas");
  }
  if (config.gs_kernel == GsKernelStructure::Classic &&
      config.panel_size != 1) {
    throw std::invalid_argument("--gs-kernel classic requires --panel 1");
  }
  if (config.snapshots > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "--snapshots exceeds the lag-offset representation");
  }
  if (config.correlation_lags > config.snapshots) {
    throw std::invalid_argument(
        "--correlation-lags must not exceed --snapshots");
  }

  std::size_t total_bytes = 0;
  if (!checked_mul(logical_bytes_per_tile(config), tile_count(config),
                   total_bytes)) {
    throw std::invalid_argument("tile count or logical state size overflows");
  }
  constexpr double gib = 1024.0 * 1024.0 * 1024.0;
  if (config.memory_limit_gib > 0.0 &&
      static_cast<double>(total_bytes) > config.memory_limit_gib * gib) {
    throw std::invalid_argument("logical state exceeds --memory-limit-gib");
  }
  return config;
}

inline std::size_t folded_index(std::size_t work_index, std::size_t size) {
  return (work_index & 1U) == 0U ? work_index / 2
                                : size - 1 - work_index / 2;
}

inline std::size_t correlation_row(std::size_t work_index, std::size_t size,
                                   CorrelationOrder order) {
  return order == CorrelationOrder::Forward
             ? work_index
             : folded_index(work_index, size);
}

inline float initial_sample(std::size_t tile, std::size_t antenna,
                            std::size_t snapshot) {
  std::uint32_t value = static_cast<std::uint32_t>(snapshot + 1);
  value ^= static_cast<std::uint32_t>((antenna + 1) * 0x9e3779b9U);
  value ^= static_cast<std::uint32_t>((tile + 7) * 0x85ebca6bU);
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  const float noise = static_cast<float>(value & 0xffffU) / 32767.5f - 1.0f;
  const float common = static_cast<float>((snapshot + 3 * tile) % 97) / 97.0f;
  const float gain = 0.85f + 0.003f * static_cast<float>(antenna % 29);
  return gain * noise + 0.12f * common +
         0.0005f * static_cast<float>(antenna);
}

inline std::vector<float> make_samples(std::size_t tile,
                                       const Config &config) {
  std::vector<float> samples(config.antennas * config.snapshots);
  for (std::size_t antenna = 0; antenna < config.antennas; ++antenna) {
    for (std::size_t snapshot = 0; snapshot < config.snapshots; ++snapshot) {
      samples[antenna * config.snapshots + snapshot] =
          initial_sample(tile, antenna, snapshot);
    }
  }
  return samples;
}

inline std::vector<float> make_initial_basis(const Config &config) {
  std::vector<float> basis(config.antennas * config.antennas, 0.0f);
  for (std::size_t row = 0; row < config.antennas; ++row) {
    for (std::size_t col = row; col < config.antennas; ++col) {
      const std::size_t distance = col - row;
      basis[row * config.antennas + col] =
          row == col ? 1.0f : 0.015f / static_cast<float>(distance + 1);
    }
  }
  return basis;
}

inline std::vector<std::uint32_t>
make_lag_offsets(std::size_t tile, const Config &config) {
  (void)tile;
  std::vector<std::uint32_t> offsets(config.correlation_lags);
  for (std::size_t index = 0; index < config.correlation_lags; ++index) {
    offsets[index] = static_cast<std::uint32_t>(index);
  }
  return offsets;
}

inline std::size_t tile_count(const Config &config) {
  std::size_t count = 0;
  if (!checked_mul(config.bands, config.time_blocks, count)) {
    throw std::overflow_error("tile count overflow");
  }
  return count;
}

inline std::size_t logical_bytes_per_tile(const Config &config) {
  std::size_t data_values = 0;
  std::size_t basis_values = 0;
  std::size_t vector_values = 0;
  std::size_t float_values = 0;
  std::size_t bytes = 0;
  std::size_t term = 0;
  if (!checked_mul(config.antennas, config.snapshots, data_values) ||
      !checked_mul(config.antennas, config.antennas, basis_values) ||
      !checked_mul(config.antennas, std::size_t{2}, vector_values) ||
      !checked_add(data_values, basis_values, float_values) ||
      !checked_add(float_values, basis_values, float_values) ||
      !checked_add(float_values, vector_values, float_values) ||
      !checked_mul(float_values, sizeof(float), bytes) ||
      !checked_mul(config.correlation_lags, sizeof(std::uint32_t), term) ||
      !checked_add(bytes, term, bytes)) {
    throw std::overflow_error("logical bytes per tile overflow");
  }
  if (!checked_mul(config.antennas, sizeof(CorrelationRowSummary), term) ||
      !checked_add(bytes, term, bytes)) {
    throw std::overflow_error("correlation summary size overflow");
  }
  if (!checked_add(bytes, sizeof(TileSummary), bytes)) {
    throw std::overflow_error("tile summary size overflow");
  }
  if (!checked_mul(config.subspace_rank, sizeof(float), term) ||
      !checked_add(bytes, term, bytes)) {
    throw std::overflow_error("MGS norm scalar array size overflow");
  }
  return bytes;
}

inline std::size_t panel_count(const Config &config) {
  return config.subspace_rank / config.panel_size +
         (config.subspace_rank % config.panel_size == 0 ? 0 : 1);
}

inline std::size_t update_count(const Config &config) {
  const std::size_t panels = panel_count(config);
  return panels == 0 ? 0 : panels - 1;
}

inline std::vector<std::size_t> active_prefix_sequence(const Config &config) {
  std::vector<std::size_t> prefixes;
  if (config.mode == Mode::CorrelationOnly) {
    return prefixes;
  }
  std::size_t panel_end = config.subspace_rank;
  while (panel_end > 0) {
    const std::size_t panel_begin =
        panel_end > config.panel_size ? panel_end - config.panel_size : 0;
    if (panel_begin > 0) {
      prefixes.push_back(panel_begin);
    }
    panel_end = panel_begin;
  }
  return prefixes;
}

inline std::size_t mgs_kernels_per_tile(const Config &config) {
  if (config.mode == Mode::CorrelationOnly) {
    return 0;
  }
  if (config.gs_kernel == GsKernelStructure::Classic) {
    std::size_t normalization_kernels = 0;
    std::size_t kernels_per_pass = 0;
    std::size_t kernels = 0;
    if (!checked_mul(config.subspace_rank, std::size_t{2},
                     normalization_kernels) ||
        !checked_add(normalization_kernels, config.subspace_rank - 1,
                     kernels_per_pass) ||
        !checked_mul(config.mgs_passes, kernels_per_pass, kernels)) {
      throw std::overflow_error("classic MGS kernel count overflows");
    }
    return kernels;
  }
  std::size_t kernels_per_pass = 0;
  std::size_t kernels = 0;
  if (!checked_add(panel_count(config), update_count(config),
                   kernels_per_pass) ||
      !checked_mul(config.mgs_passes, kernels_per_pass, kernels)) {
    throw std::overflow_error("MGS kernel count overflows");
  }
  return kernels;
}

inline std::size_t kernels_per_tile(const Config &config) {
  std::size_t kernels = 1; // final device summary
  if (config.mode != Mode::OrthogonalizeOnly) {
    kernels += 5; // mean, stddev, normalize, correlation, correlation summary
  }
  std::size_t total = 0;
  if (!checked_add(kernels, mgs_kernels_per_tile(config), total)) {
    throw std::overflow_error("kernel count per tile overflows");
  }
  return total;
}

inline work_count correlation_pair_count(const Config &config) {
  return static_cast<work_count>(config.antennas) *
         static_cast<work_count>(config.antennas + 1) / 2;
}

inline work_count correlation_work(const Config &config) {
  return correlation_pair_count(config) *
         static_cast<work_count>(config.snapshots) *
         static_cast<work_count>(config.correlation_lags);
}

inline work_count mgs_dot_update_work(const Config &config) {
  return static_cast<work_count>(config.subspace_rank) *
         static_cast<work_count>(config.subspace_rank - 1) *
         static_cast<work_count>(config.antennas) *
         static_cast<work_count>(config.mgs_passes);
}

inline work_count triangular_pairs(std::size_t begin, std::size_t end,
                                   std::size_t antennas,
                                   CorrelationOrder order) {
  work_count work = 0;
  for (std::size_t work_index = begin; work_index < end; ++work_index) {
    const std::size_t row =
        correlation_row(work_index, antennas, order);
    work += static_cast<work_count>(antennas - row);
  }
  return work;
}

inline std::vector<work_count>
theoretical_chunk_work(const Config &config) {
  std::vector<work_count> work(config.theory_devices, 0);
  if (config.mode == Mode::OrthogonalizeOnly) {
    return work;
  }
  const work_count scale = static_cast<work_count>(config.snapshots) *
                           static_cast<work_count>(config.correlation_lags);
  for (std::size_t device = 0; device < config.theory_devices; ++device) {
    const std::size_t begin = config.antennas * device / config.theory_devices;
    const std::size_t end =
        config.antennas * (device + 1) / config.theory_devices;
    work[device] = triangular_pairs(begin, end, config.antennas,
                                    config.correlation_order) *
                   scale;
  }
  return work;
}

inline void print_configuration(const Config &config, const char *runtime,
                                const char *mapper_mode) {
  const std::size_t tiles = tile_count(config);
  const std::size_t per_tile = kernels_per_tile(config);
  std::size_t total_kernels = 0;
  std::size_t total_bytes = 0;
  if (!checked_mul(per_tile, tiles, total_kernels) ||
      !checked_mul(logical_bytes_per_tile(config), tiles, total_bytes)) {
    throw std::overflow_error("reported total size overflows");
  }
  const std::vector<work_count> chunks = theoretical_chunk_work(config);
  const work_count total_correlation_work =
      config.mode == Mode::OrthogonalizeOnly ? 0 : correlation_work(config);
  const work_count total_mgs_work =
      config.mode == Mode::CorrelationOnly ? 0 : mgs_dot_update_work(config);
  const work_count matrix_elements =
      static_cast<work_count>(config.antennas) * config.antennas;
  const work_count mgs_norm_work =
      config.mode == Mode::CorrelationOnly
          ? 0
          : static_cast<work_count>(config.subspace_rank) * config.antennas *
                config.mgs_passes;
  long double max_share = 0.0L;

  std::cout << "CONFIG application=rfi_subspace_extraction"
            << " runtime=" << runtime << " antennas=" << config.antennas
            << " snapshots=" << config.snapshots
            << " bands=" << config.bands
            << " time_blocks=" << config.time_blocks
            << " tiles=" << tiles
            << " correlation_lags=" << config.correlation_lags
            << " rank=" << config.subspace_rank
            << " panel=" << config.panel_size
            << " mgs_passes=" << config.mgs_passes
            << " gs_layout=" << gs_layout_name(config.gs_layout)
            << " gs_kernel=" << gs_kernel_name(config.gs_kernel)
            << " mode=" << mode_name(config.mode)
            << " correlation_stage=" << correlation_stage_name(config)
            << " covariance_order="
            << correlation_order_name(config.correlation_order)
            << " correlation_layout="
            << correlation_layout_name(config.correlation_layout)
            << " gs_mapper=" << mapper_mode
            << " compact_gs=" << (config.compact_gs ? 1 : 0)
            << " oversubscribe=" << config.oversubscribe
            << " verify_full=" << (config.verify_full ? 1 : 0)
            << " print_manifest=" << (config.print_manifest ? 1 : 0) << '\n';
  std::cout << "DAG tiles=" << tiles
            << " submission_order=stage_major"
            << " kernels_per_tile=" << per_tile
            << " total_kernels=" << total_kernels
            << " max_width=" << tiles
            << " mgs_panels_per_pass="
            << (config.mode == Mode::CorrelationOnly ||
                        config.gs_kernel == GsKernelStructure::Classic
                    ? 0
                    : panel_count(config))
            << " mgs_norms_per_pass="
            << (config.mode != Mode::CorrelationOnly &&
                        config.gs_kernel == GsKernelStructure::Classic
                    ? config.subspace_rank
                    : 0)
            << " mgs_normalizes_per_pass="
            << (config.mode != Mode::CorrelationOnly &&
                        config.gs_kernel == GsKernelStructure::Classic
                    ? config.subspace_rank
                    : 0)
            << " mgs_updates_per_pass="
            << (config.mode == Mode::CorrelationOnly ? 0 : update_count(config))
            << " mgs_kernels_per_tile="
            << mgs_kernels_per_tile(config)
            << " active_prefix_sequence_per_pass=";
  const std::vector<std::size_t> prefixes = active_prefix_sequence(config);
  if (prefixes.empty()) {
    std::cout << "none";
  } else {
    for (std::size_t index = 0; index < prefixes.size(); ++index) {
      std::cout << (index == 0 ? "" : ",") << prefixes[index];
    }
  }
  std::cout << '\n';

  const work_count sample_elements =
      static_cast<work_count>(config.antennas) * config.snapshots;
  const work_count preprocessing_elements =
      config.mode == Mode::OrthogonalizeOnly ? 0 : sample_elements;
  std::cout << "WORK mean_sample_accumulations_per_tile="
            << work_count_string(preprocessing_elements)
            << " stddev_centered_square_accumulations_per_tile="
            << work_count_string(preprocessing_elements)
            << " normalize_elements_per_tile="
            << work_count_string(preprocessing_elements)
            << " correlation_fmas_per_tile="
            << work_count_string(total_correlation_work)
            << " correlation_fmas_batch="
            << work_count_string(total_correlation_work * tiles)
            << " correlation_snapshot_writes_per_tile="
            << work_count_string(config.mode == Mode::OrthogonalizeOnly
                                     ? 0
                                     : matrix_elements)
            << " correlation_summary_reads_per_tile="
            << work_count_string(config.mode == Mode::OrthogonalizeOnly
                                     ? 0
                                     : matrix_elements)
            << " mgs_norm_square_accumulations_per_tile="
            << work_count_string(mgs_norm_work)
            << " mgs_dot_update_fmas_per_tile="
            << work_count_string(total_mgs_work)
            << " mgs_dot_update_fmas_batch="
            << work_count_string(total_mgs_work * tiles)
            << " final_summary_basis_reads_per_tile="
            << work_count_string(matrix_elements) << '\n';
  std::cout << "MEMORY logical_bytes_per_tile="
            << logical_bytes_per_tile(config)
            << " total_logical_state_bytes=" << total_bytes << '\n';
  const long double correlation_to_mgs_ratio =
      total_mgs_work == 0
          ? 0.0L
          : static_cast<long double>(total_correlation_work) /
                static_cast<long double>(total_mgs_work);
  std::cout << "COMPOSITION correlation_to_mgs_fma_ratio=" << std::fixed
            << std::setprecision(6)
            << static_cast<double>(correlation_to_mgs_ratio)
            << " correlation_kernels_per_tile="
            << (config.mode == Mode::OrthogonalizeOnly ? 0 : 1)
            << " mgs_kernels_per_tile="
            << mgs_kernels_per_tile(config)
            << '\n';

  std::cout << "THEORY devices=" << config.theory_devices
            << " correlation_chunk_work_rows=";
  for (std::size_t device = 0; device < chunks.size(); ++device) {
    const std::size_t begin =
        config.antennas * device / config.theory_devices;
    const std::size_t end =
        config.antennas * (device + 1) / config.theory_devices;
    std::cout << (device == 0 ? "" : ",") << begin << "-" << end;
  }
  std::cout << " correlation_chunk_work=";
  for (std::size_t device = 0; device < chunks.size(); ++device) {
    std::cout << (device == 0 ? "" : ",")
              << work_count_string(chunks[device]);
  }
  std::cout << " correlation_chunk_shares=" << std::fixed
            << std::setprecision(6);
  for (std::size_t device = 0; device < chunks.size(); ++device) {
    const long double share =
        total_correlation_work == 0
            ? 0.0L
            : static_cast<long double>(chunks[device]) /
                  static_cast<long double>(total_correlation_work);
    max_share = std::max(max_share, share);
    std::cout << (device == 0 ? "" : ",")
              << static_cast<double>(share);
  }
  std::cout << " deterministic_split_speedup_ceiling="
            << static_cast<double>(max_share > 0.0L ? 1.0L / max_share : 0.0L)
            << " whole_tile_speedup_ceiling="
            << std::min(config.theory_devices, tiles) << '\n';
}

class Manifest {
public:
  explicit Manifest(const Config &config) : enabled_(config.print_manifest) {}

  void record(const Config &config, std::size_t tile, const char *stage,
              std::size_t panel_begin, std::size_t panel_end,
              std::size_t active_prefix, const std::string &global_range,
              work_count logical_read_bytes,
              work_count logical_write_bytes,
              std::size_t mgs_pass =
                  std::numeric_limits<std::size_t>::max()) {
    const std::size_t sequence = ++sequence_;
    if (!enabled_) {
      return;
    }
    std::cout << "MANIFEST sequence=" << sequence
              << " tile=" << tile
              << " time_block=" << tile / config.bands
              << " band=" << tile % config.bands
              << " stage=" << stage
              << " mgs_pass=";
    if (mgs_pass == std::numeric_limits<std::size_t>::max()) {
      std::cout << "none";
    } else {
      std::cout << mgs_pass;
    }
    std::cout
              << " panel_begin=" << panel_begin
              << " panel_end=" << panel_end
              << " active_prefix=" << active_prefix
              << " global_range=" << global_range
              << " logical_read_bytes="
              << work_count_string(logical_read_bytes)
              << " logical_write_bytes="
              << work_count_string(logical_write_bytes) << '\n';
  }

  std::size_t submitted() const { return sequence_; }

private:
  bool enabled_ = false;
  std::size_t sequence_ = 0;
};

struct VerificationResult {
  double correlation_checksum = 0.0;
  double basis_checksum = 0.0;
  double max_norm_error = 0.0;
  double max_dot = 0.0;
  double max_mask_error = 0.0;
  double min_regularized_diagonal =
      std::numeric_limits<double>::infinity();
  double min_active_correlation =
      std::numeric_limits<double>::infinity();
  double max_active_offdiagonal = 0.0;
  std::size_t correlation_nonfinite = 0;
  std::size_t basis_nonfinite = 0;
  std::size_t checked_rows = 0;
  bool passed = true;
};

inline std::vector<std::size_t> verification_rows(std::size_t rank) {
  std::vector<std::size_t> rows;
  const std::size_t count = std::min<std::size_t>(rank, 12);
  rows.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t row = count == 1 ? 0 : index * (rank - 1) / (count - 1);
    if (rows.empty() || rows.back() != row) {
      rows.push_back(row);
    }
  }
  return rows;
}

inline VerificationResult verify_summary(const Config &config,
                                         const TileSummary &summary) {
  VerificationResult result;
  result.correlation_checksum = summary.correlation_checksum;
  result.basis_checksum = summary.basis_checksum;
  result.max_norm_error = summary.max_norm_error;
  result.max_dot = summary.max_dot;
  result.max_mask_error = summary.max_mask_error;
  result.min_regularized_diagonal = summary.min_regularized_diagonal;
  result.min_active_correlation = summary.min_active_correlation;
  result.max_active_offdiagonal = summary.max_active_offdiagonal;
  result.correlation_nonfinite = summary.correlation_nonfinite;
  result.basis_nonfinite = summary.basis_nonfinite;
  result.checked_rows = summary.checked_rows;

  const bool correlation_required =
      config.mode != Mode::OrthogonalizeOnly;
  const bool correlation_ok =
      !correlation_required ||
      (summary.correlation_valid != 0 &&
       summary.correlation_nonfinite == 0 &&
       std::isfinite(summary.correlation_checksum) &&
       std::abs(summary.correlation_checksum) >= 1.0e-12 &&
       summary.max_mask_error <= 1.0e-6 &&
       summary.min_regularized_diagonal >= 0.249 &&
       summary.min_active_correlation >= -1.0e-6 &&
       (config.antennas == 1 ||
        summary.max_active_offdiagonal >= 1.0e-7));
  const bool basis_ok =
      summary.basis_nonfinite == 0 && std::isfinite(summary.basis_checksum);
  const bool orthogonality_ok =
      config.mode == Mode::CorrelationOnly ||
      (summary.max_norm_error <= 5.0e-2 && summary.max_dot <= 8.0e-2);
  result.passed = correlation_ok && basis_ok && orthogonality_ok;
  return result;
}

template <typename Reader>
inline VerificationResult verify_full_basis(const Config &config,
                                            Reader read) {
  VerificationResult result;
  const std::vector<std::size_t> rows =
      verification_rows(config.subspace_rank);
  result.checked_rows =
      config.mode == Mode::CorrelationOnly ? config.antennas : rows.size();
  for (std::size_t row = 0; row < config.antennas; ++row) {
    const std::size_t active_begin =
        correlation_row(row, config.antennas, config.correlation_order);
    for (std::size_t col = 0; col < config.antennas; ++col) {
      const double value = static_cast<double>(read(row, col));
      if (!std::isfinite(value)) {
        ++result.basis_nonfinite;
      } else {
        result.basis_checksum +=
            value * static_cast<double>(1 + (row * 17 + col * 13) % 31);
        if (config.mode == Mode::CorrelationOnly) {
          if (col < active_begin) {
            result.max_mask_error =
                std::max(result.max_mask_error, std::abs(value));
          } else {
            result.min_active_correlation =
                std::min(result.min_active_correlation, value);
            if (col > active_begin) {
              result.max_active_offdiagonal =
                  std::max(result.max_active_offdiagonal, std::abs(value));
            }
          }
          if (col == active_begin) {
            result.min_regularized_diagonal =
                std::min(result.min_regularized_diagonal, value);
          }
        }
      }
    }
  }
  if (config.mode != Mode::CorrelationOnly) {
    for (std::size_t row : rows) {
      double norm = 0.0;
      for (std::size_t col = 0; col < config.antennas; ++col) {
        const double value = static_cast<double>(
            config.gs_layout == GsLayout::ColumnVectors ? read(col, row)
                                                        : read(row, col));
        norm += value * value;
      }
      result.max_norm_error =
          std::max(result.max_norm_error, std::abs(std::sqrt(norm) - 1.0));
    }
    for (std::size_t left = 0; left < rows.size(); ++left) {
      for (std::size_t right = left + 1; right < rows.size(); ++right) {
        double dot = 0.0;
        for (std::size_t col = 0; col < config.antennas; ++col) {
          dot += static_cast<double>(
                     config.gs_layout == GsLayout::ColumnVectors
                         ? read(col, rows[left])
                         : read(rows[left], col)) *
                 static_cast<double>(
                     config.gs_layout == GsLayout::ColumnVectors
                         ? read(col, rows[right])
                         : read(rows[right], col));
        }
        result.max_dot = std::max(result.max_dot, std::abs(dot));
      }
    }
  }
  result.passed =
      result.basis_nonfinite == 0 &&
      (config.mode == Mode::CorrelationOnly
           ? result.max_mask_error <= 1.0e-6 &&
                 result.min_regularized_diagonal >= 0.249 &&
                 result.min_active_correlation >= -1.0e-6 &&
                 (config.antennas == 1 ||
                  result.max_active_offdiagonal >= 1.0e-7)
           : result.max_norm_error <= 5.0e-2 &&
                 result.max_dot <= 8.0e-2);
  return result;
}

inline void merge_verification(VerificationResult &total,
                               const VerificationResult &tile) {
  total.correlation_checksum += tile.correlation_checksum;
  total.basis_checksum += tile.basis_checksum;
  total.max_norm_error = std::max(total.max_norm_error, tile.max_norm_error);
  total.max_dot = std::max(total.max_dot, tile.max_dot);
  total.max_mask_error =
      std::max(total.max_mask_error, tile.max_mask_error);
  total.min_regularized_diagonal =
      std::min(total.min_regularized_diagonal,
               tile.min_regularized_diagonal);
  total.min_active_correlation =
      std::min(total.min_active_correlation, tile.min_active_correlation);
  total.max_active_offdiagonal =
      std::max(total.max_active_offdiagonal,
               tile.max_active_offdiagonal);
  total.correlation_nonfinite += tile.correlation_nonfinite;
  total.basis_nonfinite += tile.basis_nonfinite;
  total.checked_rows += tile.checked_rows;
  total.passed = total.passed && tile.passed;
}

} // namespace rfi_subspace_extraction
