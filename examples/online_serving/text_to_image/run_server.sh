#!/bin/bash
# Shared text-to-image online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8091}"

echo "Starting text-to-image server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
