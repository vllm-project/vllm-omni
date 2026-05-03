#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/minicpmo.yaml}"
OUTPUT="${OUTPUT:-minicpmo45_image_to_audio.wav}"
IMAGE_ARGS=()

if [[ -n "${IMAGE_PATH:-}" ]]; then
  IMAGE_ARGS=(--image-path "${IMAGE_PATH}")
fi

python examples/offline_inference/minicpmo4_5/end2end.py \
  --model-path "${MODEL}" \
  --stage-configs-path "${STAGE_CONFIG}" \
  --query-type use_image \
  --modalities audio \
  --output-wav "${OUTPUT}" \
  "${IMAGE_ARGS[@]}"
