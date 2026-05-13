#!/bin/bash
# SenseNova-U1 online serving startup script

MODEL="${MODEL:-SenseNova/SenseNova-U1-8B-MoT}"
PORT="${PORT:-8091}"

echo "Starting SenseNova-U1 server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
