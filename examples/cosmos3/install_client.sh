#!/usr/bin/env bash
set -euo pipefail
COSMOS3_INSTALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Register the optional client in the shared environment; keep the hardware's
# torch / torch-npu versions from the verified lock or existing GPU setup.
python="${COSMOS3_CLIENT_PYTHON:-${VLLM_PYTHON:-$COSMOS3_INSTALL_ROOT/.venv/bin/python}}"
[[ -x "$python" ]] || { echo "Create the shared environment first; see UNIFIED_DEPLOYMENT.md" >&2; exit 2; }
uv pip install --python "$python" --no-deps --no-build-isolation \
  --editable "$COSMOS3_INSTALL_ROOT/plugins/cosmos3_client" "$@"
