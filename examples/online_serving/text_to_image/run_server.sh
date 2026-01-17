#!/bin/bash
# Qwen-Image online serving startup script

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8091}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"

echo "Starting Qwen-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "CFG Parallel Size: $CFG_PARALLEL_SIZE"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --cfg-parallel-size "$CFG_PARALLEL_SIZE"
