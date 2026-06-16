#!/bin/bash
# Launch vLLM-Omni server for IndexTTS2
#
# Usage from repository root:
#   examples/online_serving/text_to_speech/indextts2/run_server.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8092 MODEL=/path/to/IndexTeam/IndexTTS-2 examples/online_serving/text_to_speech/indextts2/run_server.sh

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

MODEL="${MODEL:-IndexTeam/IndexTTS-2}"
PORT="${PORT:-8092}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$ROOT_DIR/vllm_omni/deploy/indextts2.yaml}"

echo "Starting IndexTTS2 server with model: $MODEL"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --omni \
    --trust-remote-code \
    --deploy-config "$DEPLOY_CONFIG"
