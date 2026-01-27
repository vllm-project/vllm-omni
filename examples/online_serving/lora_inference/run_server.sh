#!/bin/bash
# Online diffusion serving with vLLM-Omni (OpenAI-compatible API).

MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-medium}"
PORT="${PORT:-8091}"

echo "Starting vLLM-Omni diffusion server..."
echo "Model: $MODEL"
echo "Port: $PORT"

if [ -z "${VLLM_BIN:-}" ]; then
  if command -v vllm-omni >/dev/null 2>&1; then
    VLLM_BIN="vllm-omni"
  else
    VLLM_BIN="vllm"
  fi
fi

"$VLLM_BIN" serve "$MODEL" --omni \
  --port "$PORT"
