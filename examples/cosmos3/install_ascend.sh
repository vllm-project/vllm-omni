#!/usr/bin/env bash
# Rebuild a shared server/client environment from the verified offline artifacts.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifacts="${COSMOS3_ENV_ARTIFACTS:-$root/.cosmos3-env}"
env_dir="${COSMOS3_ENV_DIR:-$root/.venv}"
bootstrap="${COSMOS3_BOOTSTRAP_PYTHON:-/usr/local/python3.12.13/bin/python3}"
[[ ! -e "$env_dir" ]] || { echo "Environment already exists: $env_dir; select a new COSMOS3_ENV_DIR" >&2; exit 2; }
[[ -f "$artifacts/artifacts.sha256" ]] || { echo "Missing environment artifacts: $artifacts" >&2; exit 2; }
(cd "$artifacts" && sha256sum -c artifacts.sha256)
uv venv --system-site-packages --python "$bootstrap" "$env_dir"
uv pip install --python "$env_dir/bin/python" --no-deps --no-index \
  --find-links "$artifacts/wheels" -r "$root/examples/cosmos3/environment/ascend-py312.lock.txt"
# Only the single checkout owns application code; no private Omni source copy.
site="$env_dir/lib/python3.12/site-packages"
printf '%s\n%s\n' "$root" "$root/plugins/cosmos3_client" >"$site/cosmos3_checkout.pth"
echo "Shared server/client Python: $env_dir/bin/python"
