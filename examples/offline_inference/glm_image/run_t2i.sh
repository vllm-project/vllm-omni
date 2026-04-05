#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run GLM-Image text-to-image generation with multistage pipeline

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-/path/to/glm-image}"
CONFIG_PATH="${CONFIG_PATH:-vllm_omni/model_executor/stage_configs/glm_image.yaml}"
PROMPT="${PROMPT:-A beautiful mountain landscape with snow-capped peaks and a clear blue lake}"
OUTPUT="${OUTPUT:-output_t2i.png}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE="${GUIDANCE:-1.5}"
SEED="${SEED:-42}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "GLM-Image Text-to-Image Generation"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Prompt: ${PROMPT}"
echo "Output: ${OUTPUT}"
echo "Size: ${WIDTH}x${HEIGHT}"
echo "Steps: ${NUM_STEPS}"
echo "Guidance: ${GUIDANCE}"
echo "Seed: ${SEED}"
echo "=============================================="

python end2end.py \
    --model-path "${MODEL_PATH}" \
    --config-path "${CONFIG_PATH}" \
    --prompt "${PROMPT}" \
    --output "${OUTPUT}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-inference-steps "${NUM_STEPS}" \
    --guidance-scale "${GUIDANCE}" \
    --seed "${SEED}" \
    --verbose
