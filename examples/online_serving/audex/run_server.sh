#!/bin/bash
# Launch the vLLM-Omni server for Audex (Nemotron-Labs-Audex-2B).
#
# MODE picks the deployment (see README.md for the capability matrix):
#   tts           text -> speech            /v1/audio/speech   (default)
#   tta           caption -> general audio  /v1/audio/speech
#   thinker_only  audio -> text             /v1/chat/completions
#   s2s           cascaded speech-to-speech (both endpoints)
#
# Pass the HF repo ROOT as MODEL — per-stage subfolders resolve
# automatically.
#
# Usage:
#   ./run_server.sh                       # tts on port 8097, GPU 0
#   MODE=s2s PORT=8098 ./run_server.sh
#   MODE=tta ./run_server.sh              # needs XCodec1 (auto-downloaded)
#   MODEL=/path/to/local/snapshot ./run_server.sh

set -e

MODE="${MODE:-${1:-tts}}"
MODEL="${MODEL:-nvidia/Nemotron-Labs-Audex-2B}"
PORT="${PORT:-8097}"
GPUS="${GPUS:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEPLOY_YAML="$REPO_ROOT/vllm_omni/deploy/audex_${MODE}.yaml"
if [ ! -f "$DEPLOY_YAML" ]; then
    echo "Unknown MODE '$MODE' (no $DEPLOY_YAML); expected tts|tta|thinker_only|s2s" >&2
    exit 1
fi

echo "Starting Audex server"
echo "  MODE=$MODE ($DEPLOY_YAML)"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  CUDA_VISIBLE_DEVICES=$GPUS"

CUDA_VISIBLE_DEVICES="$GPUS" \
vllm-omni serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --stage-configs-path "$DEPLOY_YAML" \
    --omni
