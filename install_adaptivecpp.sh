#!/usr/bin/env bash
# Clones and installs AdaptiveCpp v24.10 into <build_dir>/_deps/acpp-install.
#
# Usage:
#   ./install_adaptivecpp.sh                        # NVIDIA GPU (default)
#   ./install_adaptivecpp.sh --rocm                 # AMD GPU
#   ./install_adaptivecpp.sh --cpu                  # CPU / OpenMP only
#   ./install_adaptivecpp.sh --build-dir <path>     # custom build directory (default: ./build)
#
# After installation, cmake picks up AdaptiveCpp automatically — no extra flags needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BACKEND="cuda"
JOBS="${JOBS:-$(nproc)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm)       BACKEND="rocm";    shift ;;
    --cpu)        BACKEND="cpu";     shift ;;
    --build-dir)  BUILD_DIR="$2";    shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BUILD_DIR="$(realpath -m "$BUILD_DIR")"
DEPS_DIR="${BUILD_DIR}/_deps"
PREFIX="${DEPS_DIR}/acpp-install"
SRC_DIR="${DEPS_DIR}/acpp-src"
ACPP_BUILD_DIR="${DEPS_DIR}/acpp-build"

echo "Backend  : $BACKEND"
echo "Build dir: $BUILD_DIR"
echo "Prefix   : $PREFIX"
echo "Jobs     : $JOBS"
echo ""

mkdir -p "$DEPS_DIR"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone --depth 1 --branch v24.10.0 \
    https://github.com/AdaptiveCpp/AdaptiveCpp.git "$SRC_DIR"
fi

mkdir -p "$ACPP_BUILD_DIR"
cd "$ACPP_BUILD_DIR"

case "$BACKEND" in
  cuda)
    BACKEND_FLAGS="-DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=OFF"
    ;;
  rocm)
    BACKEND_FLAGS="-DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=ON"
    ;;
  cpu)
    BACKEND_FLAGS="-DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF -DACPP_TARGETS=omp"
    ;;
esac

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DWITH_ACCELERATED_CPU=ON \
  -DWITH_SSCP_COMPILER=ON \
  -DWITH_STDPAR_COMPILER=OFF \
  -DWITH_OPENCL_BACKEND=OFF \
  $BACKEND_FLAGS \
  "$SRC_DIR"

make -j "$JOBS" install

echo ""
echo "AdaptiveCpp installed to: $PREFIX"
echo ""
echo "To build PortLBM:"
echo "  cmake -DCMAKE_CXX_COMPILER=clang++ -B ${BUILD_DIR} ${SCRIPT_DIR}"
echo "  cmake --build ${BUILD_DIR}"
