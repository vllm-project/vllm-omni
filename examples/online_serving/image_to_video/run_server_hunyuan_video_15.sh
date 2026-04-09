#!/bin/bash
# HunyuanVideo-1.5 image-to-video online serving startup script
#
# 480p: ~35 GB VRAM (BF16), fits 1x A100 80GB
# 720p: needs FP8 + VAE tiling, ~35 GB VRAM

MODEL="${MODEL:-hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v}"
PORT="${PORT:-8099}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
QUANTIZATION="${QUANTIZATION:-}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"

echo "Starting HunyuanVideo-1.5 I2V server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
echo "Quantization: ${QUANTIZATION:-none}"
echo "Cache backend: $CACHE_BACKEND"

EXTRA_FLAGS=""
if [ -n "$QUANTIZATION" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --quantization $QUANTIZATION"
fi
if [ "$CACHE_BACKEND" != "none" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --cache-backend $CACHE_BACKEND"
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --flow-shift "$FLOW_SHIFT" \
    $EXTRA_FLAGS
