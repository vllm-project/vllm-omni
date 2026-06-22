#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-GEAR-Dreams/DreamZero-DROID}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/dreamzero_tp1_cfg2.yaml}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dreamzero-droid}"

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
