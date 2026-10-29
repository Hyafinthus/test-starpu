#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${RFI_BUILD_DIR:-$SCRIPT_DIR/build}"
STARPU_PKG="${STARPU_PKG:-starpu-1.4}"
CUDA_ARCH="${CUDA_ARCH:-86}" # NVIDIA RTX A6000 (Ampere)
NVCC_BIN="${NVCC:-nvcc}"
CXX_BIN="${CXX:-g++}"
STARPU_CFLAGS_TEXT="$(pkg-config --cflags "$STARPU_PKG")"
STARPU_LIBS_TEXT="$(pkg-config --libs "$STARPU_PKG")"
read -r -a STARPU_CFLAGS <<< "$STARPU_CFLAGS_TEXT"
read -r -a STARPU_LIBS <<< "$STARPU_LIBS_TEXT"
EXTRA_FLAGS=()
if [[ -n "${NVCC_EXTRA_FLAGS:-}" ]]; then
  read -r -a EXTRA_FLAGS <<< "$NVCC_EXTRA_FLAGS"
fi
mkdir -p "$BUILD_DIR"

echo "Building with $STARPU_PKG $(pkg-config --modversion "$STARPU_PKG"), sm_$CUDA_ARCH"
"$NVCC_BIN" -O3 -DNDEBUG -std=c++17 --expt-relaxed-constexpr \
  --forward-unknown-to-host-compiler -ccbin "$CXX_BIN" \
  -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
  -gencode "arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}" \
  "${STARPU_CFLAGS[@]}" "${EXTRA_FLAGS[@]}" \
  "$SCRIPT_DIR/rfi_subspace_extraction_starpu.cpp" \
  "$SCRIPT_DIR/rfi_tasks.cpp" "$SCRIPT_DIR/rfi_cuda.cu" \
  -o "$BUILD_DIR/rfi_subspace_extraction_starpu" \
  "${STARPU_LIBS[@]}"
echo "built: $BUILD_DIR/rfi_subspace_extraction_starpu"
