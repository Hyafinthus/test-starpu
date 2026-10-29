#pragma once
#include <starpu.h>
#include "rfi_codelets.hpp"
#include "../test-scenario/rfi_subspace_extraction_common.hpp"
#include <memory>

namespace rfi_starpu {
namespace rfi = rfi_subspace_extraction;
void initialize_codelets();
void print_worker_stats();
void check_starpu(int status, const char* operation);

struct TileBuffers {
  std::size_t tile_id;
  std::vector<float> data_host, mean_host, stddev_host, basis_host;
  std::vector<float> correlation_snapshot_host;
  std::vector<std::uint32_t> lag_offsets_host;
  std::vector<rfi::CorrelationRowSummary> correlation_rows_host;
  std::vector<rfi::TileSummary> summary_host;
  std::vector<float> mgs_norm_host;
  starpu_data_handle_t data{}, mean{}, stddev{}, basis{};
  starpu_data_handle_t correlation_snapshot{}, lag_offsets{}, correlation_rows{};
  starpu_data_handle_t summary{}, mgs_norm{};

  TileBuffers(std::size_t tile, const rfi::Config& config);
  ~TileBuffers();
  TileBuffers(const TileBuffers&) = delete;
  TileBuffers& operator=(const TileBuffers&) = delete;
};

void validate_starpu_config(const rfi::Config& config);
void submit_batch(std::vector<std::unique_ptr<TileBuffers>>& tiles,
                  const rfi::Config& config, rfi::Manifest& manifest);
rfi::VerificationResult read_results(
    std::vector<std::unique_ptr<TileBuffers>>& tiles, const rfi::Config& config);
} // namespace rfi_starpu
