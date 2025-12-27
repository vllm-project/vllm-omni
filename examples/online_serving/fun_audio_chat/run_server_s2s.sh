#!/bin/bash
# Launch Fun-Audio-Chat server in S2S (Speech-to-Speech) mode
# Uses 3-stage pipeline: Main → CRQ → CosyVoice

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIGS="${SCRIPT_DIR}/../../../vllm_omni/model_executor/stage_configs/fun_audio_chat_s2s.yaml"

vllm serve FunAudioLLM/Fun-Audio-Chat-8B \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --stage-configs-path "${STAGE_CONFIGS}"
