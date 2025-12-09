#!/bin/bash
# Qwen-Image online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8091}"
NUM_GPUS="${NUM_GPUS:-1}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-4.0}"

echo "Starting Qwen-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPUs: $NUM_GPUS"

vllm serve "$MODEL" --omni \
    --port "$PORT"
