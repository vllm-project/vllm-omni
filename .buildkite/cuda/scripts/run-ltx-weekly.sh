#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

if python - <<'PY'
from kernels import get_kernel

try:
    get_kernel("shi-labs/natten", version=1)
except FileNotFoundError as exc:
    if "Cannot find a build variant for this system" not in str(exc):
        raise
    print(exc, flush=True)
    raise SystemExit(42) from exc
PY
then
    exec "$@"
else
    status=$?
    if [[ "$status" != 42 ]]; then
        exit "$status"
    fi
fi

# Remove the source build when the Hub publishes a compatible NATTEN variant.
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
source_archive="$build_dir/natten-0.21.7.tar.gz"
curl --fail --silent --show-error --location --retry 3 \
    https://files.pythonhosted.org/packages/c5/10/98fa5bc8e20937c757b5dbb2e7e949af38bc46aa92da804b0ceb07c90a86/natten-0.21.7.tar.gz \
    --output "$source_archive"
printf '%s  %s\n' 8417236b74db079b2d23899ec8f77514fb2dc73d4bbd6b346a45b6967fc05ffb "$source_archive" | sha256sum -c -
tar -xzf "$source_archive" -C "$build_dir"

# PyTorch 2.14 requires C++20. The pinned NATTEN source defaults to C++17.
source_dir="$build_dir/natten-0.21.7"
sed -i 's/set(CXX_STD "17"/set(CXX_STD "20"/; s/-std=c++17/-std=c++20/g' "$source_dir/csrc/CMakeLists.txt"
python_path=$(python -c 'import sys; print(sys.executable)')
site_packages=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
arch=$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')
if ! command -v cmake >/dev/null; then
    uv pip install --python "$python_path" --target "$build_dir/tools" cmake==4.4.0
    export PATH="$build_dir/tools/bin:$PATH" PYTHONPATH="$build_dir/tools${PYTHONPATH:+:$PYTHONPATH}"
fi
CPATH="$site_packages/nvidia/cu13/include:$site_packages/nvidia/cusparselt/include${CPATH:+:$CPATH}" \
    CUDACXX="${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}" CMAKE_GENERATOR=Ninja \
    NATTEN_CUDA_ARCH="$arch" NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-8}" \
    uv build --wheel --no-build-isolation --python "$python_path" --out-dir "$build_dir/wheels" "$source_dir"
uv pip install --python "$python_path" --no-deps --target "$build_dir/site" "$build_dir"/wheels/natten-*.whl

# The kernels loader uses this override for both na3d and token permutation.
kernel_dir="$build_dir/kernel"
mkdir "$kernel_dir"
cat > "$kernel_dir/metadata.json" <<EOF
{"name":"natten","id":"ltx_weekly_natten","version":1,"license":"MIT","backend":{"type":"cuda","archs":["$arch"]},"python-depends":[]}
EOF
cat > "$kernel_dir/__init__.py" <<'PY'
from natten import HAS_LIBNATTEN, na3d, token_permute

if not HAS_LIBNATTEN:
    raise ImportError("NATTEN was built without its native CUDA extension")
PY
export PYTHONPATH="$build_dir/site${PYTHONPATH:+:$PYTHONPATH}"
export LOCAL_KERNELS="${LOCAL_KERNELS:+$LOCAL_KERNELS:}shi-labs/natten=$kernel_dir"
"$@"
