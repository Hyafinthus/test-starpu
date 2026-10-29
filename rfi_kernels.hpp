#pragma once

// Computation bodies ported from rfi_subspace_extraction_sycl_timer.cpp.
#include "rfi_codelets.hpp"
#include <cmath>

namespace rfi_starpu {
namespace rfi = rfi_subspace_extraction;

struct Index {
  std::size_t row, col = 0;
  __host__ __device__ std::size_t operator[](unsigned axis) const { return axis ? col : row; }
};

struct Matrix {
  float* ptr;
  std::size_t ld;
  __host__ __device__ float& operator[](Index index) const {
    return ptr[index.row * ld + index.col];
  }
};

__device__ inline bool finite(double value) {
  return isfinite(value);
}

__device__ inline float inverse_sqrt(float value) {
  return rsqrtf(value);
}

constexpr std::size_t panel_threads = 128;

__device__ inline void mean_item(
    Matrix data, float* mean,
    const KernelArgs& args, Index item) {
  const auto snapshots = args.snapshots;
  const std::size_t antenna = item[0];
  float sum = 0.0f;
  for (std::size_t sample = 0; sample < snapshots; ++sample) {
    sum += data[{antenna, sample}];
  }
  mean[antenna] = sum / static_cast<float>(snapshots);
}

__device__ inline void stddev_item(
    Matrix data, const float* mean, float* stddev,
    const KernelArgs& args, Index item) {
  const auto snapshots = args.snapshots;
  const std::size_t antenna = item[0];
  float sum = 0.0f;
  for (std::size_t sample = 0; sample < snapshots; ++sample) {
    const float centered = data[{antenna, sample}] - mean[antenna];
    sum += centered * centered;
  }
  const float variance = sum / static_cast<float>(snapshots);
  stddev[antenna] = sqrtf(fmaxf(variance, 1.0e-12f));
}

__device__ inline void normalize_item(
    Matrix data, const float* mean, const float* stddev,
    const KernelArgs& args, Index item) {
  const std::size_t antenna = item[0];
  data[item] = (data[item] - mean[antenna]) / stddev[antenna];
}

__device__ inline void row_correlation_item(
    Matrix data, const std::uint32_t* lag_offsets, Matrix basis, Matrix correlation_snapshot,
    const KernelArgs& args, Index item) {
  const auto antennas = args.antennas;
  const auto snapshots = args.snapshots;
  const auto correlation_lags = args.correlation_lags;
  const auto folded = args.folded;
  const std::size_t work_row = item[0];
  const std::size_t antenna0 =
      folded ? ((work_row & 1U) == 0U
                    ? work_row / 2
                    : antennas - 1 - work_row / 2)
             : work_row;
  for (std::size_t antenna1 = 0; antenna1 < antennas; ++antenna1) {
    if (antenna1 < antenna0) {
      basis[{work_row, antenna1}] = 0.0f;
      correlation_snapshot[{work_row, antenna1}] = 0.0f;
      continue;
    }
    float correlation_energy = 0.0f;
    for (std::size_t lag_index = 0;
         lag_index < correlation_lags; ++lag_index) {
      std::size_t shifted_sample = lag_offsets[lag_index];
      float dot = 0.0f;
      for (std::size_t sample = 0; sample < snapshots; ++sample) {
        dot += data[{antenna0, sample}] *
               data[{antenna1, shifted_sample}];
        shifted_sample =
            shifted_sample + 1 == snapshots ? 0 : shifted_sample + 1;
      }
      const float lag_correlation =
          dot / static_cast<float>(snapshots);
      correlation_energy += lag_correlation * lag_correlation;
    }
    float correlation = sqrtf(
        correlation_energy / static_cast<float>(correlation_lags));
    if (antenna1 == antenna0) {
      correlation += 0.25f;
    }
    basis[{work_row, antenna1}] = correlation;
    correlation_snapshot[{work_row, antenna1}] = correlation;
  }
}

__device__ inline void pair_correlation_item(
    Matrix data, const std::uint32_t* lag_offsets, Matrix basis, Matrix correlation_snapshot,
    const KernelArgs& args, Index item) {
  const auto antennas = args.antennas;
  const auto snapshots = args.snapshots;
  const auto correlation_lags = args.correlation_lags;
  const auto folded = args.folded;
  const std::size_t work_row = item[0];
  const std::size_t antenna1 = item[1];
  const std::size_t antenna0 =
      folded ? ((work_row & 1U) == 0U
                    ? work_row / 2
                    : antennas - 1 - work_row / 2)
             : work_row;
  if (antenna1 < antenna0) {
    basis[item] = 0.0f;
    correlation_snapshot[item] = 0.0f;
    return;
  }
  float correlation_energy = 0.0f;
  for (std::size_t lag_index = 0;
       lag_index < correlation_lags; ++lag_index) {
    std::size_t shifted_sample = lag_offsets[lag_index];
    float dot = 0.0f;
    for (std::size_t sample = 0; sample < snapshots; ++sample) {
      dot += data[{antenna0, sample}] *
             data[{antenna1, shifted_sample}];
      shifted_sample =
          shifted_sample + 1 == snapshots ? 0 : shifted_sample + 1;
    }
    const float lag_correlation =
        dot / static_cast<float>(snapshots);
    correlation_energy += lag_correlation * lag_correlation;
  }
  float correlation = sqrtf(
      correlation_energy / static_cast<float>(correlation_lags));
  if (antenna1 == antenna0) {
    correlation += 0.25f;
  }
  basis[item] = correlation;
  correlation_snapshot[item] = correlation;
}

__device__ inline void correlation_summary_item(
    Matrix correlation_snapshot, rfi::CorrelationRowSummary* rows,
    const KernelArgs& args, Index item) {
  const auto antennas = args.antennas;
  const auto folded = args.folded;
  const std::size_t work_row = item[0];
  const std::size_t active_begin =
      folded ? ((work_row & 1U) == 0U
                    ? work_row / 2
                    : antennas - 1 - work_row / 2)
             : work_row;
  rfi::CorrelationRowSummary output;
  output.checksum = 0.0;
  output.min_regularized_diagonal =
      std::numeric_limits<double>::infinity();
  output.min_active_correlation =
      std::numeric_limits<double>::infinity();
  output.max_active_offdiagonal = 0.0;
  output.max_mask_error = 0.0;
  output.nonfinite = 0;
  for (std::size_t col = 0; col < antennas; ++col) {
    const double value =
        static_cast<double>(correlation_snapshot[{work_row, col}]);
    if (!finite(value)) {
      ++output.nonfinite;
      continue;
    }
    output.checksum +=
        value * static_cast<double>(
                    1 + (work_row * 17 + col * 13) % 31);
    if (col < active_begin) {
      output.max_mask_error =
          fmax(output.max_mask_error, fabs(value));
    } else {
      output.min_active_correlation =
          fmin(output.min_active_correlation, value);
      if (col > active_begin) {
        output.max_active_offdiagonal =
            fmax(output.max_active_offdiagonal,
                       fabs(value));
      }
    }
    if (col == active_begin) {
      output.min_regularized_diagonal =
          fmin(output.min_regularized_diagonal, value);
    }
  }
  rows[work_row] = output;
}

__device__ inline void reverse_norm_item(
    Matrix basis, float* norm,
    const KernelArgs& args, Index item) {
  const auto antennas = args.antennas;
  const auto column_vectors = args.column_vectors;
  const auto pivot = args.panel_begin;
  float norm_squared = 0.0f;
  for (std::size_t element = 0; element < antennas; ++element) {
    const float value =
        column_vectors ? basis[{element, pivot}]
                       : basis[{pivot, element}];
    norm_squared += value * value;
  }
  norm[pivot] =
      sqrtf(fmaxf(norm_squared, 1.0e-20f));
}

__device__ inline void reverse_normalize_item(
    Matrix basis, const float* norm,
    const KernelArgs& args, Index item) {
  const auto column_vectors = args.column_vectors;
  const auto pivot = args.panel_begin;
  const std::size_t element = item[0];
  if (column_vectors) {
    basis[{element, pivot}] /= norm[pivot];
  } else {
    basis[{pivot, element}] /= norm[pivot];
  }
}

template <typename Team>
__device__ inline void reverse_panel_item(
    Matrix basis, float* scratch,
    const KernelArgs& args, Team item) {
  const auto antennas = args.antennas;
  const auto column_vectors = args.column_vectors;
  const auto panel_begin = args.panel_begin;
  const auto panel_end = args.panel_end;
  constexpr std::size_t local_size = panel_threads;
  const std::size_t lane = item.get_local_linear_id();
  for (std::size_t pivot_cursor = panel_end;
       pivot_cursor > panel_begin; --pivot_cursor) {
    const std::size_t pivot = pivot_cursor - 1;
    float norm_squared = 0.0f;
    for (std::size_t col = lane; col < antennas;
         col += local_size) {
      const float value =
          column_vectors ? basis[{col, pivot}]
                         : basis[{pivot, col}];
      norm_squared += value * value;
    }
    scratch[lane] = norm_squared;
    item.barrier();
    for (std::size_t stride = local_size / 2; stride > 0;
         stride /= 2) {
      if (lane < stride) {
        scratch[lane] += scratch[lane + stride];
      }
      item.barrier();
    }
    const float inverse_norm =
        inverse_sqrt(fmaxf(scratch[0], 1.0e-20f));
    for (std::size_t col = lane; col < antennas;
         col += local_size) {
      if (column_vectors) {
        basis[{col, pivot}] *= inverse_norm;
      } else {
        basis[{pivot, col}] *= inverse_norm;
      }
    }
    item.barrier();
    for (std::size_t row = panel_begin; row < pivot; ++row) {
      float projection = 0.0f;
      for (std::size_t col = lane; col < antennas;
           col += local_size) {
        projection +=
            (column_vectors ? basis[{col, row}]
                            : basis[{row, col}]) *
            (column_vectors ? basis[{col, pivot}]
                            : basis[{pivot, col}]);
      }
      scratch[lane] = projection;
      item.barrier();
      for (std::size_t stride = local_size / 2; stride > 0;
           stride /= 2) {
        if (lane < stride) {
          scratch[lane] += scratch[lane + stride];
        }
        item.barrier();
      }
      projection = scratch[0];
      for (std::size_t col = lane; col < antennas;
           col += local_size) {
        if (column_vectors) {
          basis[{col, row}] -= projection * basis[{col, pivot}];
        } else {
          basis[{row, col}] -= projection * basis[{pivot, col}];
        }
      }
      item.barrier();
    }
  }
}

__device__ inline void reverse_update_item(
    Matrix basis,
    const KernelArgs& args, Index item) {
  const auto antennas = args.antennas;
  const auto column_vectors = args.column_vectors;
  const auto panel_begin = args.panel_begin;
  const auto panel_end = args.panel_end;
  const std::size_t row = item[0];
  if (row >= panel_begin) {
    return;
  }
  for (std::size_t pivot_cursor = panel_end;
       pivot_cursor > panel_begin; --pivot_cursor) {
    const std::size_t pivot = pivot_cursor - 1;
    float projection = 0.0f;
    for (std::size_t col = 0; col < antennas; ++col) {
      projection +=
          (column_vectors ? basis[{col, row}] : basis[{row, col}]) *
          (column_vectors ? basis[{col, pivot}]
                          : basis[{pivot, col}]);
    }
    for (std::size_t col = 0; col < antennas; ++col) {
      if (column_vectors) {
        basis[{col, row}] -= projection * basis[{col, pivot}];
      } else {
        basis[{row, col}] -= projection * basis[{pivot, col}];
      }
    }
  }
}

__device__ inline void final_summary_item(
    Matrix basis, const rfi::CorrelationRowSummary* correlation_rows, rfi::TileSummary* summary,
    const KernelArgs& args, Index item) {
  const auto antennas = args.antennas;
  const auto rank = args.rank;
  const auto has_correlation = args.has_correlation;
  const auto check_orthogonality = args.check_orthogonality;
  const auto column_vectors = args.column_vectors;
  rfi::TileSummary output;
  output.correlation_checksum = 0.0;
  output.basis_checksum = 0.0;
  output.max_norm_error = 0.0;
  output.max_dot = 0.0;
  output.max_mask_error = 0.0;
  output.min_regularized_diagonal =
      has_correlation
          ? std::numeric_limits<double>::infinity()
          : 0.0;
  output.min_active_correlation =
      has_correlation
          ? std::numeric_limits<double>::infinity()
          : 0.0;
  output.max_active_offdiagonal = 0.0;
  output.correlation_nonfinite = 0;
  output.basis_nonfinite = 0;
  output.checked_rows = 0;
  output.correlation_valid = has_correlation ? 1 : 0;

  if (has_correlation) {
    for (std::size_t row = 0; row < antennas; ++row) {
      const rfi::CorrelationRowSummary current =
          correlation_rows[row];
      output.correlation_checksum += current.checksum;
      output.correlation_nonfinite += current.nonfinite;
      output.max_mask_error =
          fmax(output.max_mask_error,
                     current.max_mask_error);
      output.min_regularized_diagonal =
          fmin(output.min_regularized_diagonal,
                     current.min_regularized_diagonal);
      output.min_active_correlation =
          fmin(output.min_active_correlation,
                     current.min_active_correlation);
      output.max_active_offdiagonal =
          fmax(output.max_active_offdiagonal,
                     current.max_active_offdiagonal);
    }
  }

  for (std::size_t row = 0; row < antennas; ++row) {
    for (std::size_t col = 0; col < antennas; ++col) {
      const double value =
          static_cast<double>(basis[{row, col}]);
      if (!finite(value)) {
        ++output.basis_nonfinite;
      } else {
        output.basis_checksum +=
            value * static_cast<double>(
                        1 + (row * 17 + col * 13) % 31);
      }
    }
  }

  if (check_orthogonality) {
    const std::size_t row_count = rank < 12 ? rank : 12;
    output.checked_rows =
        static_cast<std::uint32_t>(row_count);
    for (std::size_t index = 0; index < row_count; ++index) {
      const std::size_t row =
          row_count == 1 ? 0 : index * (rank - 1) / (row_count - 1);
      double norm = 0.0;
      for (std::size_t col = 0; col < antennas; ++col) {
        const double value = static_cast<double>(
            column_vectors ? basis[{col, row}]
                           : basis[{row, col}]);
        norm += value * value;
      }
      output.max_norm_error =
          fmax(output.max_norm_error,
                     fabs(sqrt(norm) - 1.0));
    }
    for (std::size_t left = 0; left < row_count; ++left) {
      const std::size_t left_row =
          row_count == 1 ? 0
                         : left * (rank - 1) / (row_count - 1);
      for (std::size_t right = left + 1; right < row_count;
           ++right) {
        const std::size_t right_row =
            right * (rank - 1) / (row_count - 1);
        double dot = 0.0;
        for (std::size_t col = 0; col < antennas; ++col) {
          dot += static_cast<double>(
                     column_vectors ? basis[{col, left_row}]
                                    : basis[{left_row, col}]) *
                 static_cast<double>(
                     column_vectors ? basis[{col, right_row}]
                                    : basis[{right_row, col}]);
        }
        output.max_dot =
            fmax(output.max_dot, fabs(dot));
      }
    }
  } else {
    output.checked_rows =
        static_cast<std::uint32_t>(antennas);
  }
  summary[0] = output;
}

} // namespace rfi_starpu

