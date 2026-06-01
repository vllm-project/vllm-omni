#!/bin/bash
# Launch vLLM-Omni server for OmniVoice TTS
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_server.sh

set -e

MODEL="${MODEL:-k2-fsa/OmniVoice}"
PORT="${PORT:-8000}"

echo "Starting OmniVoice server with model: $MODEL"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni
