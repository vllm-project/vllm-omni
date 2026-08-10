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
#   ./run_server.sh                       # 2B tts on port 8097, GPU 0
#   SIZE=30b ./run_server.sh              # 30B-A3B tts (explicit 30B yaml)
#   SIZE=30b MODE=s2s PORT=8098 ./run_server.sh
#   MODE=tta ./run_server.sh              # needs XCodec1 (auto-downloaded)
#   MODEL=/path/to/local/snapshot ./run_server.sh

set -e

MODE="${MODE:-${1:-tts}}"
SIZE="${SIZE:-2b}"
if [ "$SIZE" = "30b" ]; then
    MODEL="${MODEL:-nvidia/Nemotron-Labs-Audex-30B-A3B}"
    YAML_SUFFIX="_30b"
elif [ "$SIZE" = "2b" ]; then
    MODEL="${MODEL:-nvidia/Nemotron-Labs-Audex-2B}"
    YAML_SUFFIX=""
else
    echo "Unknown SIZE '$SIZE'; expected 2b|30b" >&2
    exit 1
fi
PORT="${PORT:-8097}"
GPUS="${GPUS:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEPLOY_YAML="$REPO_ROOT/vllm_omni/deploy/audex_${MODE}${YAML_SUFFIX}.yaml"
if [ ! -f "$DEPLOY_YAML" ]; then
    echo "Unknown MODE '$MODE' / SIZE '$SIZE' (no $DEPLOY_YAML); expected MODE=tts|tta|thinker_only|s2s, SIZE=2b|30b" >&2
    exit 1
fi

echo "Starting Audex server"
echo "  MODE=$MODE ($DEPLOY_YAML)"
echo "  SIZE=$SIZE"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  CUDA_VISIBLE_DEVICES=$GPUS"

CUDA_VISIBLE_DEVICES="$GPUS" \
vllm-omni serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --deploy-config "$DEPLOY_YAML" \
    --omni
