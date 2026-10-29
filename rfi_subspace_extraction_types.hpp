#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace rfi_subspace_extraction {

struct CorrelationRowSummary {
  double checksum = 0.0;
  double min_regularized_diagonal =
      std::numeric_limits<double>::infinity();
  double min_active_correlation =
      std::numeric_limits<double>::infinity();
  double max_active_offdiagonal = 0.0;
  double max_mask_error = 0.0;
  std::uint64_t nonfinite = 0;
};

struct TileSummary {
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
  std::uint64_t correlation_nonfinite = 0;
  std::uint64_t basis_nonfinite = 0;
  std::uint32_t checked_rows = 0;
  std::uint32_t correlation_valid = 0;
};

static_assert(std::is_trivially_copyable<CorrelationRowSummary>::value,
              "CorrelationRowSummary must be trivially copyable");
static_assert(std::is_trivially_copyable<TileSummary>::value,
              "TileSummary must be trivially copyable");

} // namespace rfi_subspace_extraction
