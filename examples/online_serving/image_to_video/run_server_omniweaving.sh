#!/bin/bash
# OmniWeaving image-to-video online serving startup script

set -euo pipefail

MODEL="${MODEL:-Tencent-Hunyuan/OmniWeaving}"
PORT="${PORT:-8096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"

echo "Starting OmniWeaving I2V server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Flow shift: $FLOW_SHIFT"

VLLM_NO_DEEP_GEMM=1 TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --flow-shift "$FLOW_SHIFT"
