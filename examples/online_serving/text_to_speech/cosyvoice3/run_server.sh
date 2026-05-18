#!/bin/bash
# Launch vLLM-Omni server for CosyVoice3 online serving.
#
# Usage:
#   ./run_server.sh        # async_chunk mode (default)
#   ./run_server.sh async
#   ./run_server.sh sync   # legacy full-sequence code2wav path
#
# Environment overrides:
#   COSYVOICE3_MODEL=/path/or/hf-repo
#   COSYVOICE3_TOKENIZER=/path/to/CosyVoice-BlankEN
#   COSYVOICE3_PORT=8091
#   COSYVOICE3_DEPLOY_CONFIG=/path/to/cosyvoice3.yaml
#   VLLM_OMNI_BIN=vllm-omni

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

MODE="${1:-async}"
VLLM_OMNI_BIN="${VLLM_OMNI_BIN:-vllm-omni}"
MODEL="${COSYVOICE3_MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
TOKENIZER="${COSYVOICE3_TOKENIZER:-${MODEL}/CosyVoice-BlankEN}"
PORT="${COSYVOICE3_PORT:-8091}"
DEPLOY_CONFIG="${COSYVOICE3_DEPLOY_CONFIG:-${REPO_ROOT}/vllm_omni/deploy/cosyvoice3.yaml}"
EXTRA_ARGS=()

case "${MODE}" in
    async|async_chunk)
        ;;
    sync)
        EXTRA_ARGS+=("--no-async-chunk")
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Supported modes: async, async_chunk, sync"
        exit 1
        ;;
esac

echo "Starting CosyVoice3 server"
echo "  model: ${MODEL}"
echo "  tokenizer: ${TOKENIZER}"
echo "  mode: ${MODE}"
echo "  port: ${PORT}"

"${VLLM_OMNI_BIN}" serve "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --deploy-config "${DEPLOY_CONFIG}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --stage-init-timeout 900 \
    --init-timeout 1200 \
    --omni \
    "${EXTRA_ARGS[@]}"
