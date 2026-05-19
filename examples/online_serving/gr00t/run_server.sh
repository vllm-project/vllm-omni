#!/usr/bin/env bash
set -euo pipefail

# Launch the GR00T-N1.7 OpenPI policy server.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_server.sh
#
# Overridable via env vars: MODEL, HOST, PORT, DEPLOY_CONFIG,
# SERVED_MODEL_NAME, ATTENTION_BACKEND, DIFFUSION_ATTENTION_BACKEND.

MODEL="${MODEL:-nvidia/GR00T-N1.7-3B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# Resolve the deploy YAML against the repo root so the script can be run
# from any cwd (e.g. examples/online_serving/gr00t/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${REPO_ROOT}/vllm_omni/deploy/gr00t.yaml}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gr00t-n17}"

args=(
  serve
  "$MODEL"
  --omni
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --deploy-config "$DEPLOY_CONFIG"
  --enforce-eager
  --disable-log-stats
)

ATTENTION_BACKEND="${ATTENTION_BACKEND:-torch}" \
DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-TORCH_SDPA}" \
vllm "${args[@]}"
