#!/bin/bash
# Launch HyperCLOVAX-SEED-Omni-8B with vLLM-Omni.
#
# Requirements:
#   - 6× GPUs (≥24 GB VRAM each):
#       GPU 0-3: Thinker (tensor_parallel_size=4)
#       GPU 4  : Vision decoder
#       GPU 5  : Audio decoder
#   - HF model: naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B
#
# Usage:
#   ./run_server.sh [--model MODEL] [--port PORT] [--stage-configs-path PATH]

set -e

MODEL="${MODEL:-naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
STAGE_CONFIG="${STAGE_CONFIG:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STAGE_CONFIG="$SCRIPT_DIR/../../../vllm_omni/model_executor/stage_configs/hcx_omni.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL="$2";       shift 2 ;;
        --port)          PORT="$2";        shift 2 ;;
        --host)          HOST="$2";        shift 2 ;;
        --stage-configs-path) STAGE_CONFIG="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--model MODEL] [--port PORT] [--host HOST] [--stage-configs-path PATH]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$STAGE_CONFIG" ]] && STAGE_CONFIG="$DEFAULT_STAGE_CONFIG"

echo "================================================="
echo " HyperCLOVAX-SEED-Omni-8B  vLLM-Omni Server"
echo "================================================="
echo " Model       : $MODEL"
echo " Stage config: $STAGE_CONFIG"
echo " Endpoint    : http://$HOST:$PORT/v1"
echo "================================================="

python -m vllm_omni.entrypoints.openai.api_server \
    --model "$MODEL" \
    --stage-configs-path "$STAGE_CONFIG" \
    --port "$PORT" \
    --host "$HOST" \
    --trust-remote-code
