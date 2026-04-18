#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh                                                        # Default: 1.7B CustomVoice
#   ./run_server.sh CustomVoice                                            # 1.7B CustomVoice
#   ./run_server.sh VoiceDesign 1.7B                                       # 1.7B VoiceDesign
#   ./run_server.sh Base 0.6B                                              # 0.6B Base (voice clone)
#   ./run_server.sh /path/to/local/model                                   # Local model directory
#   SERVED_MODEL_NAME=my-tts ./run_server.sh                               # Custom served model name
#   STAGE_CONFIGS_PATH=./my_qwen_stage_config.yaml ./run_server.sh         # Custom stage config

set -e

ARG="${1:-CustomVoice}"
SIZE="${2:-1.7B}"

if [ -d "$ARG" ]; then
    MODEL="$ARG"
else
    TASK_TYPE="$ARG"
    case "$SIZE" in
        0.6B|1.7B) ;;
        *)
            echo "Unknown size: $SIZE"
            echo "Supported: 0.6B, 1.7B"
            exit 1
            ;;
    esac
    VALID_MODELS=(
        "1.7B-CustomVoice"
        "1.7B-VoiceDesign"
        "1.7B-Base"
        "0.6B-CustomVoice"
        "0.6B-Base"
    )
    COMBO="${SIZE}-${TASK_TYPE}"
    MATCHED=false
    for valid in "${VALID_MODELS[@]}"; do
        if [ "$COMBO" = "$valid" ]; then
            MATCHED=true
            break
        fi
    done
    if [ "$MATCHED" = false ]; then
        echo "Error: Invalid model combination: Qwen/Qwen3-TTS-12Hz-${COMBO}"
        echo ""
        echo "Valid models:"
        for valid in "${VALID_MODELS[@]}"; do
            echo "  Qwen/Qwen3-TTS-12Hz-${valid}"
        done
        exit 1
    fi
    MODEL="Qwen/Qwen3-TTS-12Hz-${COMBO}"
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
STAGE_CONFIGS_PATH="${STAGE_CONFIGS_PATH:-vllm_omni/model_executor/stage_configs/qwen3_tts.yaml}"

echo "Starting Qwen3-TTS server with model: $MODEL"

SERVE_ARGS=(
    --stage-configs-path "$STAGE_CONFIGS_PATH"
    --host 0.0.0.0
    --port 8091
    --gpu-memory-utilization 0.9
    --trust-remote-code
    --omni
)

if [ -n "$SERVED_MODEL_NAME" ]; then
    echo "Served model name: $SERVED_MODEL_NAME"
    SERVE_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

vllm-omni serve "$MODEL" "${SERVE_ARGS[@]}"
