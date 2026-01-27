#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh                           # Default: CustomVoice model
#   ./run_server.sh CustomVoice               # CustomVoice model
#   ./run_server.sh VoiceDesign               # VoiceDesign model
#   ./run_server.sh Base                      # Base (voice clone) model

set -e

TASK_TYPE="${1:-CustomVoice}"

case "$TASK_TYPE" in
    CustomVoice)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ;;
    VoiceDesign)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ;;
    Base)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ;;
    *)
        echo "Unknown task type: $TASK_TYPE"
        echo "Supported: CustomVoice, VoiceDesign, Base"
        exit 1
        ;;
esac

echo "Starting Qwen3-TTS server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni
