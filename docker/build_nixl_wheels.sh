#!/usr/bin/env bash
# Build the NIXL native and meta wheels with the UCX plugin enabled.
set -euo pipefail

NIXL_VERSION="${1:-v1.3.0}"
OUTPUT_DIR="${2:-$PWD/dist}"
UCX_PREFIX="${3:-/opt/venv}"
BUILD_ROOT="$(mktemp -d)"
SOURCE_DIR="${BUILD_ROOT}/nixl"

cleanup() {
  rm -rf "${BUILD_ROOT}"
}
trap cleanup EXIT

mkdir -p "${OUTPUT_DIR}"

git clone --depth 1 --branch "${NIXL_VERSION}" https://github.com/ai-dynamo/nixl.git "${SOURCE_DIR}"
cd "${SOURCE_DIR}"

uv pip install \
  build \
  meson \
  meson-python \
  ninja \
  patchelf \
  pybind11 \
  pyyaml \
  tomlkit

export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${UCX_PREFIX}/lib/ucx:${LD_LIBRARY_PATH:-}"

uv build \
  --wheel \
  --no-build-isolation \
  --out-dir "${OUTPUT_DIR}" \
  --config-setting setup-args="-Ducx_path=${UCX_PREFIX}" \
  --config-setting setup-args="-Denable_plugins=UCX" \
  --config-setting setup-args="-Ddisable_gds_backend=true"

meson setup meta-build \
  -Ducx_path="${UCX_PREFIX}" \
  -Denable_plugins=UCX \
  -Ddisable_gds_backend=true
meta_target="$(
  ninja -C meta-build -t targets all |
    sed -n 's/: CUSTOM_COMMAND$//p' |
    grep -E 'nixl-[^-]+-py3-none-any\.whl$' |
    head -n 1
)"
[[ -n "${meta_target}" ]] || { echo "NIXL meta wheel target was not found" >&2; exit 1; }
ninja -C meta-build "${meta_target}"
cp meta-build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl "${OUTPUT_DIR}/"

native_wheel="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'nixl_cu12-*.whl' -print -quit)"
meta_wheel="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'nixl-*.whl' ! -name 'nixl_cu12-*.whl' -print -quit)"
[[ -n "${native_wheel}" ]] || { echo "NIXL native wheel was not produced" >&2; exit 1; }
[[ -n "${meta_wheel}" ]] || { echo "NIXL meta wheel was not produced" >&2; exit 1; }

printf '%s\n%s\n' "${native_wheel}" "${meta_wheel}"
