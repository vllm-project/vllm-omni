#!/bin/bash
# Wan2.2 image-to-video server start script

MODEL="${MODEL:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
PORT="${PORT:-8099}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CACHE_DIT_SUMMARY="${ENABLE_CACHE_DIT_SUMMARY:-0}"

echo "Starting Wan2.2 I2V server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Cache backend: $CACHE_BACKEND"
if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then
    echo "Cache-DiT summary: enabled"
fi

CACHE_BACKEND_FLAG=""
if [ "$CACHE_BACKEND" != "none" ]; then
    CACHE_BACKEND_FLAG="--cache-backend $CACHE_BACKEND"
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    $CACHE_BACKEND_FLAG \
    $(if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then echo "--enable-cache-dit-summary"; fi)
