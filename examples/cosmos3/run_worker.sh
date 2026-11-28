#!/usr/bin/env bash
# Optional direct worker entry point, for running the frontend separately.
set -euo pipefail
if [[ "${1:-}" == --device ]]; then
  export COSMOS3_DEVICE="${2:?--device requires cuda or npu}"
  shift 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VLLM_OMNI_ROOT"
exec "$VLLM_PYTHON" -u -m vllm_omni.entrypoints.cli.main \
  serve "${COSMOS3_MODEL:-$VLLM_OMNI_ROOT/models/cosmos3-edge-libero10-diffusers}" \
  --omni --host 127.0.0.1 --port "${COSMOS3_WORKER_PORT:-8100}" \
  --init-timeout 1800 --no-guardrails "${COSMOS3_WORKER_ARGS[@]}" "$@"
