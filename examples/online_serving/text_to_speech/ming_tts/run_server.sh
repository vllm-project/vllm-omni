#!/bin/bash
# Launch vLLM-Omni server for Ming-omni-tts.
#
# Usage:
#   ./run_server.sh
#   PORT=8000 ./run_server.sh

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../../.." && pwd)"

MODEL="${MODEL:-inclusionAI/Ming-omni-tts-0.5B}"
PORT="${PORT:-8091}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$ROOT/vllm_omni/deploy/ming_tts.yaml}"

echo "Starting Ming-omni-tts server with model: $MODEL"
echo "Deploy config: $DEPLOY_CONFIG"

vllm-omni serve "$MODEL" \
    --deploy-config "$DEPLOY_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --enforce-eager \
    --omni
