#!/bin/bash
# Launch vLLM-Omni server for Ming-flash-omni-2.0 standalone talker (TTS).
#
# Usage:
#   ./run_server.sh
#   MODEL=/path/to/local/model ./run_server.sh
#   PORT=8091 ./run_server.sh
#   HOST=127.0.0.1 ./run_server.sh   # bind only to loopback

set -e

MODEL="${MODEL:-Jonathan1909/Ming-flash-omni-2.0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/ming_flash_omni_tts.yaml}"

echo "Starting Ming standalone TTS server with model: $MODEL"
echo "Deploy config: $DEPLOY_CONFIG"

vllm serve "$MODEL" \
    --deploy-config "$DEPLOY_CONFIG" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --omni
