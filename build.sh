#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${RFI_BUILD_DIR:-$SCRIPT_DIR/build}"
STARPU_PKG="${STARPU_PKG:-starpu-1.4}"
CUDA_ARCH="${CUDA_ARCH:-75}" # CUDA 10.1 PTX; A6000 uses driver JIT.
NVCC_BIN="${NVCC:-nvcc}"
CXX_BIN="${CXX:-g++}"

parse_flags() {
  local words word
  words="$(python3 -c 'import shlex, sys; print("\n".join(shlex.split(sys.argv[1])))' "$1")"
  PARSED_FLAGS=()
  if [[ -n "$words" ]]; then
    while IFS= read -r word; do
      PARSED_FLAGS+=("$word")
    done <<< "$words"
  fi
}

STARPU_CFLAGS_TEXT="$(pkg-config --cflags "$STARPU_PKG")"
STARPU_LIBS_TEXT="$(pkg-config --libs "$STARPU_PKG")"
parse_flags "$STARPU_CFLAGS_TEXT"
STARPU_CFLAGS=("${PARSED_FLAGS[@]}")
parse_flags "$STARPU_LIBS_TEXT"
STARPU_LIBS=("${PARSED_FLAGS[@]}")
NVCC_CFLAGS=()
for flag in "${STARPU_CFLAGS[@]}"; do
  case "$flag" in
    -I*|-D*|-U*) NVCC_CFLAGS+=("$flag") ;;
    *) NVCC_CFLAGS+=(-Xcompiler "$flag") ;;
  esac
done
NVCC_LIBS=()
for flag in "${STARPU_LIBS[@]}"; do
  case "$flag" in
    -Wl,*) NVCC_LIBS+=(-Xlinker "${flag#-Wl,}") ;;
    -l*|-L*|*.a|*.so) NVCC_LIBS+=("$flag") ;;
    *) NVCC_LIBS+=(-Xcompiler "$flag") ;;
  esac
done
EXTRA_FLAGS=()
if [[ -n "${NVCC_EXTRA_FLAGS:-}" ]]; then
  parse_flags "$NVCC_EXTRA_FLAGS"
  EXTRA_FLAGS=("${PARSED_FLAGS[@]}")
fi
mkdir -p "$BUILD_DIR"

echo "Building with $STARPU_PKG $(pkg-config --modversion "$STARPU_PKG"), C++14, compute_$CUDA_ARCH PTX"
"$CXX_BIN" -O3 -DNDEBUG -std=c++14 "${STARPU_CFLAGS[@]}" \
  -c "$SCRIPT_DIR/rfi_subspace_extraction_starpu.cpp" \
  -o "$BUILD_DIR/rfi_subspace_extraction_starpu.o"
"$CXX_BIN" -O3 -DNDEBUG -std=c++14 "${STARPU_CFLAGS[@]}" \
  -c "$SCRIPT_DIR/rfi_tasks.cpp" -o "$BUILD_DIR/rfi_tasks.o"

"$NVCC_BIN" -O3 -DNDEBUG -std=c++14 --expt-relaxed-constexpr \
  -ccbin "$CXX_BIN" \
  -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
  -gencode "arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}" \
  "${NVCC_CFLAGS[@]}" "${EXTRA_FLAGS[@]}" \
  -c "$SCRIPT_DIR/rfi_cuda.cu" -o "$BUILD_DIR/rfi_cuda.o"

"$NVCC_BIN" -ccbin "$CXX_BIN" --cudart shared \
  "$BUILD_DIR/rfi_subspace_extraction_starpu.o" \
  "$BUILD_DIR/rfi_tasks.o" "$BUILD_DIR/rfi_cuda.o" \
  -o "$BUILD_DIR/rfi_subspace_extraction_starpu" \
  "${NVCC_LIBS[@]}"
echo "built: $BUILD_DIR/rfi_subspace_extraction_starpu"
