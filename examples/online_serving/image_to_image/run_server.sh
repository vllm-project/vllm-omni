#!/bin/bash
# Qwen-Image-Edit online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image-Edit}"
PORT="${PORT:-8092}"

echo "Starting Qwen-Image-Edit server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
