#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
PORT="${PORT:-8091}"
HOST="${HOST:-0.0.0.0}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/minicpmo.yaml}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-vllm_omni/model_executor/models/minicpmo4_5/chat_template.jinja}"

if command -v vllm-omni >/dev/null 2>&1; then
  SERVER_CLI=(vllm-omni)
else
  SERVER_CLI=(python -m vllm_omni.entrypoints.cli.main)
fi

"${SERVER_CLI[@]}" serve "${MODEL}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --stage-configs-path "${STAGE_CONFIG}" \
  --chat-template "${CHAT_TEMPLATE}" \
  --chat-template-content-format openai \
  --trust-remote-code
