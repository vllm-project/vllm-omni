#!/bin/bash
# Launch vLLM-Omni server for FunCineForge (movie dubbing & TTS).
#
# Usage:
#   ./run_server.sh
#   MODEL=/local/path/Fun-CineForge PORT=8092 ./run_server.sh

set -e

MODEL="${MODEL:-FunAudioLLM/Fun-CineForge}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/funcineforge.yaml}"

echo "Starting FunCineForge server with model: $MODEL"
echo "Deploy config: $DEPLOY_CONFIG"

vllm serve "$MODEL" \
    --deploy-config "$DEPLOY_CONFIG" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --omni
