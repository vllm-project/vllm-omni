#!/bin/bash
# SenseNova-U1 online serving startup script

MODEL="${MODEL:-SenseNova/SenseNova-U1-8B-MoT}"
PORT="${PORT:-8091}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"

echo "Starting SenseNova-U1 server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Cache backend: $CACHE_BACKEND"

EXTRA_ARGS=()
if [ "$CACHE_BACKEND" != "none" ]; then
    EXTRA_ARGS+=(--cache-backend "$CACHE_BACKEND")
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    "${EXTRA_ARGS[@]}"
