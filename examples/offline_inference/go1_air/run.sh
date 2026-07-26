#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GO1_AIR_MODEL_DIR is optional; smoke test runs in stub mode without it.
if [[ -n "${GO1_AIR_MODEL_DIR:-}" ]]; then
  export GO1_AIR_MODEL_DIR
fi

python "$ROOT_DIR/smoke.py" "$@"
