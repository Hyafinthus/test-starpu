#pragma once

#include "rfi_subspace_extraction_types.hpp"
#include <cstddef>

namespace rfi_starpu {
namespace rfi = rfi_subspace_extraction;

enum class Stage {
  Mean,
  Stddev,
  Normalize,
  RowCorrelation,
  PairCorrelation,
  CorrelationSummary,
  ReverseNorm,
  ReverseNormalize,
  ReversePanel,
  ReverseUpdate,
  FinalSummary,
  Tile,
  Count
};

struct KernelArgs {
  std::size_t antennas, snapshots, correlation_lags, rank;
  std::size_t panel_begin = 0, panel_end = 0, execution_rows = 0;
  std::size_t row_begin = 0, row_count = 0;
  std::size_t panel_size = 1, mgs_passes = 1, mgs_pass = 0;
  bool pair_parallel = true, classic = false, compact = false;
  bool folded, column_vectors, has_correlation, check_orthogonality;
};
static_assert(std::is_trivially_copyable<KernelArgs>::value,
              "KernelArgs must be trivially copyable");

using CudaFunction = void (*)(void*[], void*);
CudaFunction cuda_function(Stage stage);
void record_cuda_kernel();
} // namespace rfi_starpu
