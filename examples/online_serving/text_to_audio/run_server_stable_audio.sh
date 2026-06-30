#!/bin/bash
# Stable Audio Open online serving startup script.
# Serves text-to-audio generation through the OpenAI-compatible
# `POST /v1/audio/generate` endpoint (see run_curl_stable_audio.sh).

MODEL="${MODEL:-stabilityai/stable-audio-open-1.0}"
PORT="${PORT:-8091}"

echo "Starting Stable Audio Open server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager
