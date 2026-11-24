#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VLLM_OMNI_ROOT"
exec "$VLLM_PYTHON" -u -m vllm_omni.entrypoints.cosmos3.policy_server \
  --vllm-base-url "http://127.0.0.1:${COSMOS3_WORKER_PORT:-8100}" \
  --port "${COSMOS3_POLICY_PORT:-8101}" --flow-shift 1.0 "$@"
