#!/bin/bash
# Launch vLLM-Omni server for CosyVoice3 TTS
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_server.sh
#
# Streaming (async-chunk) is on by default via vllm_omni/deploy/cosyvoice3.yaml.
# Set NO_ASYNC_CHUNK=1 to use the legacy synchronous path.

set -e

MODEL="${MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
PORT="${PORT:-8091}"

EXTRA_ARGS=()
if [[ -n "${NO_ASYNC_CHUNK:-}" ]]; then
    EXTRA_ARGS+=(--no-async-chunk)
fi

echo "Starting CosyVoice3 server with model: $MODEL"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni \
    "${EXTRA_ARGS[@]}"
