#pragma once

#include "../test-scenario/rfi_subspace_extraction_types.hpp"
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
  Count
};

struct KernelArgs {
  std::size_t antennas, snapshots, correlation_lags, rank;
  std::size_t panel_begin = 0, panel_end = 0, execution_rows = 0;
  bool folded, column_vectors, has_correlation, check_orthogonality;
};
static_assert(std::is_trivially_copyable<KernelArgs>::value,
              "KernelArgs must be trivially copyable");

using CudaFunction = void (*)(void*[], void*);
CudaFunction cuda_function(Stage stage);
void record_cuda_task();
} // namespace rfi_starpu
