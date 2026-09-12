#!/bin/bash
# Wan2.2 online serving startup script

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8098}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
CACHE_CONFIG="${CACHE_CONFIG:-}"
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

EXTRA_ARGS=()
if [ "$CACHE_BACKEND" != "none" ]; then
    EXTRA_ARGS+=(--cache-backend "$CACHE_BACKEND")
    if [ -n "$CACHE_CONFIG" ]; then
        EXTRA_ARGS+=(--cache-config "$CACHE_CONFIG")
    fi
fi
if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then
    EXTRA_ARGS+=(--enable-cache-dit-summary)
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --boundary-ratio "$BOUNDARY_RATIO" \
    --flow-shift "$FLOW_SHIFT" \
    "${EXTRA_ARGS[@]}"
