#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run GLM-Image image-to-image (editing) with multistage pipeline

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-/path/to/glm-image}"
CONFIG_PATH="${CONFIG_PATH:-vllm_omni/model_executor/stage_configs/glm_image.yaml}"
PROMPT="${PROMPT:-Transform this image into an oil painting style}"
INPUT_IMAGE=""
OUTPUT="${OUTPUT:-output_i2i.png}"
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
        --image)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
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

# Check if input image is provided
if [ -z "${INPUT_IMAGE}" ]; then
    echo "Error: --image is required for image-to-image mode"
    echo "Usage: ./run_i2i.sh --image /path/to/input.png [--prompt \"edit instruction\"]"
    exit 1
fi

if [ ! -f "${INPUT_IMAGE}" ]; then
    echo "Error: Input image not found: ${INPUT_IMAGE}"
    exit 1
fi

echo "=============================================="
echo "GLM-Image Image-to-Image Generation"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Input: ${INPUT_IMAGE}"
echo "Prompt: ${PROMPT}"
echo "Output: ${OUTPUT}"
echo "Steps: ${NUM_STEPS}"
echo "Guidance: ${GUIDANCE}"
echo "Seed: ${SEED}"
echo "=============================================="

python end2end.py \
    --model-path "${MODEL_PATH}" \
    --config-path "${CONFIG_PATH}" \
    --prompt "${PROMPT}" \
    --image "${INPUT_IMAGE}" \
    --output "${OUTPUT}" \
    --num-inference-steps "${NUM_STEPS}" \
    --guidance-scale "${GUIDANCE}" \
    --seed "${SEED}" \
    --verbose
