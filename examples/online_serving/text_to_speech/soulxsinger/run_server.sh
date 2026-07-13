#!/bin/bash
# Launch vLLM-Omni server for SoulX-Singer (single-stage DiT, preprocess inline).
#
# Usage:
#   MODEL=/path/to/SoulX-Singer PREPROCESS=/path/to/Preprocess \
#
# Audio paths in client extra_args must be readable on the server host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODEL="${MODEL:-Soul-AILab/SoulX-Singer}"
MODE="${MODE:-svs}"
PORT="${PORT:-8192}"
GPUS="${GPUS:-0}"

if [[ "$MODE" == "svc" ]]; then
  DEPLOY_CONFIG="${DEPLOY_CONFIG:-$REPO_ROOT/vllm_omni/deploy/soulxsinger_svc.yaml}"
else
  DEPLOY_CONFIG="${DEPLOY_CONFIG:-$REPO_ROOT/vllm_omni/deploy/soulxsinger_svs.yaml}"
fi

echo "Starting SoulX-Singer server"
echo "  MODEL=$MODEL"
echo "  MODE=$MODE"
echo "  PORT=$PORT"
echo "  DEPLOY_CONFIG=$DEPLOY_CONFIG"
echo "  CUDA_VISIBLE_DEVICES=$GPUS"

CUDA_VISIBLE_DEVICES="$GPUS" \
vllm serve "$MODEL" \
    --omni \
    --deploy-config "$DEPLOY_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --enforce-eager
