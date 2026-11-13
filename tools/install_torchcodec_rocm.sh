#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Build TorchCodec against the ROCm PyTorch installation.  The PyPI wheel is
# built against upstream PyTorch and is not ABI-compatible with ROCm builds.

set -euo pipefail

TORCHCODEC_REPO="${TORCHCODEC_REPO:-https://github.com/pytorch/torchcodec.git}"
# Keep this pinned for ROCm/PyTorch ABI compatibility. When rebasing the ROCm
# base image or vLLM, revalidate this version against the active torch build.
TORCHCODEC_BRANCH="${TORCHCODEC_BRANCH:-v0.10.0}"
TORCHCODEC_WHEEL_CACHE="${TORCHCODEC_WHEEL_CACHE:-/root/.cache/torchcodec-wheels}"

echo "=== TorchCodec ROCm installation ==="

verify_torchcodec() {
    python3 -c "from torchcodec.decoders import VideoDecoder"
}

if verify_torchcodec >/dev/null 2>&1; then
    echo "TorchCodec is already installed and working."
    exit 0
fi

install_system_deps() {
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "TorchCodec build dependencies are missing and apt-get is unavailable."
        return 1
    fi
    apt-get update
    apt-get install -y --no-install-recommends \
        pkg-config \
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
        libswscale-dev libavdevice-dev libavfilter-dev libswresample-dev
}

if ! command -v pkg-config >/dev/null 2>&1 || \
   ! pkg-config --exists libavcodec libavformat libavutil libswscale \
       libavdevice libavfilter libswresample 2>/dev/null; then
    echo "Installing missing FFmpeg development dependencies..."
    install_system_deps
fi

if ! python3 -c "import packaging, pybind11" >/dev/null 2>&1; then
    python3 -m pip install packaging pybind11 setuptools wheel
fi

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
export pybind11_DIR="$PYBIND11_DIR"
export CMAKE_PREFIX_PATH="${PYBIND11_DIR}:${CMAKE_PREFIX_PATH:-}"

safe_component() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

PYTHON_SOABI=$(python3 -c \
    "import sysconfig; print(sysconfig.get_config_var('SOABI') or 'unknown')")
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
ROCM_VERSION=$(python3 -c "import torch; print(torch.version.hip or 'none')")
FFMPEG_VERSION=$(pkg-config --modversion libavcodec 2>/dev/null || echo unknown)
ARCH_TAG="${PYTORCH_ROCM_ARCH:-all}"

CACHE_DIR="${TORCHCODEC_WHEEL_CACHE}/$(safe_component "$TORCHCODEC_BRANCH")/$(safe_component "$PYTHON_SOABI")/$(safe_component "$TORCH_VERSION")/$(safe_component "$ROCM_VERSION")/$(safe_component "$FFMPEG_VERSION")/$(safe_component "$ARCH_TAG")"
CACHED_WHEEL=""
if [ -d "$CACHE_DIR" ]; then
    CACHED_WHEEL=$(find "$CACHE_DIR" -maxdepth 1 -type f -name 'torchcodec-*.whl' -print -quit)
fi

validate_wheel_filename() {
    WHEEL_PATH="$1" python3 - <<'PY'
from os import environ
from pathlib import Path

from packaging.utils import parse_wheel_filename

wheel = Path(environ["WHEEL_PATH"])
parse_wheel_filename(wheel.name)
PY
}

if [ -n "$CACHED_WHEEL" ] && validate_wheel_filename "$CACHED_WHEEL"; then
    echo "Trying cached wheel: $CACHED_WHEEL"
    if python3 -m pip install --no-deps --force-reinstall "$CACHED_WHEEL" && \
       verify_torchcodec; then
        echo "Installed TorchCodec from cache."
        exit 0
    fi
    echo "Cached wheel is incompatible; rebuilding TorchCodec."
    rm -f "$CACHED_WHEEL"
fi

BUILD_DIR=$(mktemp -d -t torchcodec-XXXXXX)
trap 'rm -rf "$BUILD_DIR"' EXIT

cd "$BUILD_DIR"
echo "Cloning TorchCodec $TORCHCODEC_BRANCH..."
git clone --depth 1 --branch "$TORCHCODEC_BRANCH" "$TORCHCODEC_REPO" torchcodec
cd torchcodec

export TORCHCODEC_CMAKE_BUILD_DIR="${PWD}/build"
export TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=1
export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
export CMAKE_GENERATOR=Ninja
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

if command -v ccache >/dev/null 2>&1; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

echo "Building TorchCodec (MAX_JOBS=$MAX_JOBS)..."
python3 -m pip wheel . --no-build-isolation --no-deps -w "$BUILD_DIR/dist"
BUILT_WHEEL=$(find "$BUILD_DIR/dist" -maxdepth 1 -type f -name 'torchcodec-*.whl' -print -quit)
test -n "$BUILT_WHEEL"
validate_wheel_filename "$BUILT_WHEEL"

python3 -m pip install --no-deps --force-reinstall "$BUILT_WHEEL"
verify_torchcodec

mkdir -p "$CACHE_DIR"
cp "$BUILT_WHEEL" "$CACHE_DIR/$(basename "$BUILT_WHEEL")"
echo "Cached TorchCodec wheel in $CACHE_DIR"
echo "=== TorchCodec ROCm installation complete ==="
