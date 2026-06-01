#!/bin/bash
# Launch vLLM-Omni server for MOSS-TTS-Nano
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8000 ./run_server.sh

set -e

MODEL="${MODEL:-OpenMOSS-Team/MOSS-TTS-Nano}"
PORT="${PORT:-8000}"

echo "Starting MOSS-TTS-Nano server with model: $MODEL"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --omni
