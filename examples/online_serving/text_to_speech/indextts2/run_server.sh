#!/bin/bash
# Launch vLLM-Omni server for IndexTTS 2.0 or 2.5.
#
# Usage from repository root:
#   examples/online_serving/text_to_speech/indextts2/run_server.sh
#   MODEL_VERSION=2.5 MODEL=/path/to/native/bundle \
#     examples/online_serving/text_to_speech/indextts2/run_server.sh

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

MODEL_VERSION="${MODEL_VERSION:-2.0}"
PORT="${PORT:-8092}"

if [[ "$MODEL_VERSION" == "2.5" ]]; then
    if [[ -z "${MODEL:-}" ]]; then
        echo "Usage: MODEL_VERSION=2.5 MODEL=/path/to/native/bundle bash $0" >&2
        exit 2
    fi
    DEFAULT_DEPLOY_CONFIG="$ROOT_DIR/vllm_omni/deploy/indextts2_5.yaml"
elif [[ "$MODEL_VERSION" == "2.0" ]]; then
    MODEL="${MODEL:-IndexTeam/IndexTTS-2}"
    DEFAULT_DEPLOY_CONFIG="$ROOT_DIR/vllm_omni/deploy/indextts2.yaml"
else
    echo "MODEL_VERSION must be 2.0 or 2.5" >&2
    exit 2
fi

DEPLOY_CONFIG="${DEPLOY_CONFIG:-$DEFAULT_DEPLOY_CONFIG}"

echo "Starting IndexTTS $MODEL_VERSION server with model: $MODEL"
echo "Deploy config: $DEPLOY_CONFIG"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --omni \
    --trust-remote-code \
    --deploy-config "$DEPLOY_CONFIG"
