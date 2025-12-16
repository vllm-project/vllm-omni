#!/bin/bash
# Qwen-Image online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8091}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "Starting Qwen-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPUs: $NUM_GPUS"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --num-gpus "$NUM_GPUS"
