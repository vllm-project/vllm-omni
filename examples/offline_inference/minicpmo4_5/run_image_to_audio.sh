#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/minicpmo4_5.yaml}"
OUTPUT="${OUTPUT:-minicpmo45_image_to_audio.wav}"
IMAGE_ARGS=()

if [[ -n "${IMAGE_PATH:-}" ]]; then
  IMAGE_ARGS=(--image-path "${IMAGE_PATH}")
fi

python examples/offline_inference/minicpmo4_5/end2end.py \
  --model-path "${MODEL}" \
  --deploy-config "${DEPLOY_CONFIG}" \
  --query-type use_image \
  --modalities audio \
  --output-wav "${OUTPUT}" \
  "${IMAGE_ARGS[@]}"
