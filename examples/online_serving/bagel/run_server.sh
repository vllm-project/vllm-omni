#!/bin/bash
# Bagel online serving startup script

MODEL="${MODEL:-ByteDance-Seed/BAGEL-7B-MoT}"
PORT="${PORT:-8091}"

echo "Starting Bagel server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
