#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/models/nava}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nava}"

exec vllm-omni serve "${MODEL}" \
  --omni \
  --model-class-name NAVAPipeline \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}"
