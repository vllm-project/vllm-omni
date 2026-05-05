#!/bin/bash
# Launch vLLM-Omni server for Fish Speech S2 Pro
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_server.sh

set -e

MODEL="${MODEL:-fishaudio/s2-pro}"
PORT="${PORT:-8091}"

echo "Starting Fish Speech S2 Pro server with model: $MODEL"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve "$MODEL" \
    --omni \
    --host 0.0.0.0 \
    --port "$PORT"
