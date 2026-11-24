#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Restore the verified native runtime and rebuild an empty venv; no Docker use.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifacts="${COSMOS3_NATIVE_ARTIFACTS:-$root/.cosmos3-native}"
env_dir="${COSMOS3_ENV_DIR:-$root/.venv-native}"
uv_bin="${COSMOS3_UV:-$artifacts/bin/uv}"
[[ -x "$uv_bin" ]] || { echo "Missing uv bootstrap: $uv_bin" >&2; exit 2; }
[[ ! -e "$env_dir" ]] || { echo "Environment exists: $env_dir; choose COSMOS3_ENV_DIR" >&2; exit 2; }
(cd "$artifacts" && sha256sum -c artifacts.sha256)
if [[ ! -d "$root/.runtime/cann-9.1.0" ]]; then
  tar -xf "$artifacts/native-runtime.tar" -C "$root"
fi
[[ -f "$artifacts/mesa-23.2.1.tar" ]] || { echo "Missing pinned Mesa artifact: $artifacts/mesa-23.2.1.tar" >&2; exit 2; }
if [[ ! -f "$root/.runtime/mesa-23.2.1/lib/libOSMesa.so" ]]; then
  tar -xf "$artifacts/mesa-23.2.1.tar" -C "$root"
fi
python="${COSMOS3_BOOTSTRAP_PYTHON:-$root/.runtime/python/bin/python3.12}"
"$uv_bin" venv --offline --python "$python" "$env_dir"
site="$env_dir/lib/python3.12/site-packages"
printf '%s\n%s\n%s\n' "$root" "$root/plugins/cosmos3_client" "$root/.runtime/image-site" > "$site/cosmos3_native.pth"
"$uv_bin" pip install --python "$env_dir/bin/python" --no-deps --no-index \
  --find-links "$artifacts/wheels" -r "$root/examples/cosmos3/environment/native-overlay.lock.txt"
SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0rc1 VLLM_OMNI_TARGET_DEVICE=npu \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$uv_bin" pip install \
  --python "$env_dir/bin/python" --offline --no-deps --no-build-isolation -e "$root"
echo "Native environment ready: $env_dir/bin/python"
echo "source $root/examples/cosmos3/env_native.sh"
