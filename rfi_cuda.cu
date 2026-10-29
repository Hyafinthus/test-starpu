#include <cuda_runtime.h>
#include <starpu_cuda.h>
#include <starpu.h>
#include <algorithm>
#include "rfi_codelets.hpp"
#include "rfi_kernels.hpp"

namespace rfi_starpu {
namespace {
constexpr unsigned block_threads = 256;

Matrix matrix(void* buffer) {
  return {reinterpret_cast<float*>(STARPU_MATRIX_GET_PTR(buffer)),
          STARPU_MATRIX_GET_LD(buffer)};
}
template <typename T> T* vector(void* buffer) {
  return reinterpret_cast<T*>(STARPU_VECTOR_GET_PTR(buffer));
}
unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>(
      (count + block_threads - 1) / block_threads, 65535));
}
struct CudaTeam {
  __device__ std::size_t get_local_linear_id() const { return threadIdx.x; }
  __device__ void barrier() const { __syncthreads(); }
};
void check_launch() {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) STARPU_CUDA_REPORT_ERROR(error);
  record_cuda_task();
}

__global__ void mean_kernel(Matrix data, float* mean, KernelArgs args) {
  const std::size_t count = args.antennas;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    mean_item(data, mean, args, Index{i});
  }
}

void mean_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  mean_kernel<<<blocks(args.antennas), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<float>(buffers[1]), args);
  check_launch();
}

__global__ void stddev_kernel(Matrix data, const float* mean, float* stddev, KernelArgs args) {
  const std::size_t count = args.antennas;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    stddev_item(data, mean, stddev, args, Index{i});
  }
}

void stddev_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  stddev_kernel<<<blocks(args.antennas), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<float>(buffers[1]), vector<float>(buffers[2]), args);
  check_launch();
}

__global__ void normalize_kernel(Matrix data, const float* mean, const float* stddev, KernelArgs args) {
  const std::size_t count = args.antennas * args.snapshots;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    normalize_item(data, mean, stddev, args, Index{i / args.snapshots, i % args.snapshots});
  }
}

void normalize_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  normalize_kernel<<<blocks(args.antennas * args.snapshots), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<float>(buffers[1]), vector<float>(buffers[2]), args);
  check_launch();
}

__global__ void row_correlation_kernel(Matrix data, const std::uint32_t* lag_offsets, Matrix basis, Matrix correlation_snapshot, KernelArgs args) {
  const std::size_t count = args.antennas;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    row_correlation_item(data, lag_offsets, basis, correlation_snapshot, args, Index{i});
  }
}

void row_correlation_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  row_correlation_kernel<<<blocks(args.antennas), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<std::uint32_t>(buffers[1]), matrix(buffers[2]), matrix(buffers[3]), args);
  check_launch();
}

__global__ void pair_correlation_kernel(Matrix data, const std::uint32_t* lag_offsets, Matrix basis, Matrix correlation_snapshot, KernelArgs args) {
  const std::size_t count = args.antennas * args.antennas;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    pair_correlation_item(data, lag_offsets, basis, correlation_snapshot, args, Index{i / args.antennas, i % args.antennas});
  }
}

void pair_correlation_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  pair_correlation_kernel<<<blocks(args.antennas * args.antennas), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<std::uint32_t>(buffers[1]), matrix(buffers[2]), matrix(buffers[3]), args);
  check_launch();
}

__global__ void correlation_summary_kernel(Matrix correlation_snapshot, rfi::CorrelationRowSummary* rows, KernelArgs args) {
  const std::size_t count = args.antennas;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    correlation_summary_item(correlation_snapshot, rows, args, Index{i});
  }
}

void correlation_summary_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  correlation_summary_kernel<<<blocks(args.antennas), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<rfi::CorrelationRowSummary>(buffers[1]), args);
  check_launch();
}

__global__ void reverse_norm_kernel(Matrix basis, float* norm, KernelArgs args) {
  reverse_norm_item(basis, norm, args, Index{0});
}

void reverse_norm_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  reverse_norm_kernel<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<float>(buffers[1]), args);
  check_launch();
}

__global__ void reverse_normalize_kernel(Matrix basis, const float* norm, KernelArgs args) {
  const std::size_t count = args.antennas;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    reverse_normalize_item(basis, norm, args, Index{i});
  }
}

void reverse_normalize_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  reverse_normalize_kernel<<<blocks(args.antennas), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<float>(buffers[1]), args);
  check_launch();
}

__global__ void reverse_panel_kernel(Matrix basis, KernelArgs args) {
  __shared__ float scratch[panel_threads];
  reverse_panel_item(basis, scratch, args, CudaTeam{});
}

void reverse_panel_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  reverse_panel_kernel<<<1, panel_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), args);
  check_launch();
}

__global__ void reverse_update_kernel(Matrix basis, KernelArgs args) {
  const std::size_t count = args.execution_rows;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(blockDim.x) * gridDim.x) {
    reverse_update_item(basis, args, Index{i});
  }
}

void reverse_update_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  reverse_update_kernel<<<blocks(args.execution_rows), block_threads, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), args);
  check_launch();
}

__global__ void final_summary_kernel(Matrix basis, const rfi::CorrelationRowSummary* correlation_rows, rfi::TileSummary* summary, KernelArgs args) {
  final_summary_item(basis, correlation_rows, summary, args, Index{0});
}

void final_summary_cuda(void* buffers[], void* cl_arg) {
  KernelArgs args;
  starpu_codelet_unpack_args(cl_arg, &args);
  final_summary_kernel<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(
      matrix(buffers[0]), vector<rfi::CorrelationRowSummary>(buffers[1]), vector<rfi::TileSummary>(buffers[2]), args);
  check_launch();
}

} // namespace

CudaFunction cuda_function(Stage stage) {
  static const CudaFunction functions[] = {
      mean_cuda,
      stddev_cuda,
      normalize_cuda,
      row_correlation_cuda,
      pair_correlation_cuda,
      correlation_summary_cuda,
      reverse_norm_cuda,
      reverse_normalize_cuda,
      reverse_panel_cuda,
      reverse_update_cuda,
      final_summary_cuda,
  };
  static_assert(sizeof(functions) / sizeof(functions[0]) ==
                static_cast<std::size_t>(Stage::Count), "CUDA codelet count");
  return functions[static_cast<std::size_t>(stage)];
}
} // namespace rfi_starpu
