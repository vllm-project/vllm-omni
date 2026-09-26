#!/bin/bash
# Launch vLLM-Omni server for GLM-TTS models
#
# Usage:
#   ./run_server.sh                           # Default model path, async_chunk mode
#   ./run_server.sh /path/to/GLM-TTS          # Custom model path, async_chunk mode
#   ./run_server.sh /path/to/GLM-TTS sync     # Sync two-stage mode
#
# NOTE: The model path should point to the repo ROOT (not llm/ subdirectory).
# model_subdir/tokenizer_subdir in the pipeline config resolve subdirectories.

set -e

MODEL="${1:-zai-org/GLM-TTS}"
MODE="${2:-async}"

EXTRA_ARGS=()
case "$MODE" in
    async|async_chunk)
        ;;
    sync|no_async_chunk)
        EXTRA_ARGS+=("--no-async-chunk")
        ;;
    *)
        echo "Unknown mode: $MODE (expected async or sync)" >&2
        exit 1
        ;;
esac

echo "Starting GLM-TTS server with model: $MODEL (mode: $MODE)"

vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/glm_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni \
    "${EXTRA_ARGS[@]}"
