#!/bin/bash
# Launch vLLM-Omni server for IndexTTS2
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8092 MODEL=/path/to/IndexTeam/IndexTTS-2 ./run_server.sh

set -e

MODEL="${MODEL:-IndexTeam/IndexTTS-2}"
PORT="${PORT:-8092}"

echo "Starting IndexTTS2 server with model: $MODEL"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --omni
