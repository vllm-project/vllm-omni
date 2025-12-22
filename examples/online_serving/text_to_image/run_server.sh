#!/bin/bash
# Qwen-Image online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8091}"

echo "Starting Qwen-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
