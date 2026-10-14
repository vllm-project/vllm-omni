#!/bin/bash
# Wan2.2 S2V (speech-to-video) online serving startup script

MODEL="${MODEL:-Wan-AI/Wan2.2-S2V-14B}"
PORT="${PORT:-8091}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
TP="${TP:-2}"
CACHE_BACKEND="${CACHE_BACKEND:-cache_dit}"

echo "Starting Wan2.2 S2V server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
echo "Tensor parallel size: $TP"
echo "Cache backend: $CACHE_BACKEND"

CACHE_BACKEND_FLAG=""
if [ "$CACHE_BACKEND" != "none" ]; then
    CACHE_BACKEND_FLAG="--cache-backend $CACHE_BACKEND"
fi

VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$MODEL" --omni \
    --model-class-name WanS2VPipeline \
    --tensor-parallel-size "$TP" \
    --flow-shift "$FLOW_SHIFT" \
    --vae-use-slicing --vae-use-tiling \
    --port "$PORT" \
    $CACHE_BACKEND_FLAG
