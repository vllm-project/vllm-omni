#!/bin/bash
# Qwen-Image-Edit online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image-Edit}"
PORT="${PORT:-8092}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "Starting Qwen-Image-Edit server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPUs: $NUM_GPUS"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --num-gpus "$NUM_GPUS"
