#!/bin/bash
# Wan2.2 online serving startup script

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8098}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CACHE_DIT_SUMMARY="${ENABLE_CACHE_DIT_SUMMARY:-0}"

echo "Starting Wan2.2 server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Boundary ratio: $BOUNDARY_RATIO"
echo "Flow shift: $FLOW_SHIFT"
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
    --boundary-ratio "$BOUNDARY_RATIO" \
    --flow-shift "$FLOW_SHIFT" \
    $CACHE_BACKEND_FLAG \
    $(if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then echo "--enable-cache-dit-summary"; fi)
