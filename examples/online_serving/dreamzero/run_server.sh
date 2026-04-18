#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-GEAR-Dreams/DreamZero-DROID}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dreamzero-droid}"

args=(
  serve
  "$MODEL"
  --omni
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --enforce-eager
  --disable-log-stats
)

if [[ -n "$CFG_PARALLEL_SIZE" ]]; then
  args+=(--cfg-parallel-size "$CFG_PARALLEL_SIZE")
fi

ATTENTION_BACKEND="${ATTENTION_BACKEND:-torch}" \
DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-TORCH_SDPA}" \
vllm "${args[@]}"
