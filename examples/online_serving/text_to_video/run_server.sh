#!/bin/bash
# Wan2.2 text-to-video server

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8093}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"

echo "Starting server for: $MODEL"
echo "Port: $PORT"
echo "boundary_ratio: $BOUNDARY_RATIO"
echo "flow_shift: $FLOW_SHIFT"

vllm serve "$MODEL" --omni --port "$PORT" --boundary-ratio "$BOUNDARY_RATIO" --flow-shift "$FLOW_SHIFT"
