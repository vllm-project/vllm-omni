#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/minicpmo.yaml}"
OUTPUT="${OUTPUT:-minicpmo45_video_to_audio.wav}"
NUM_FRAMES="${NUM_FRAMES:-30}"
VIDEO_ARGS=()

if [[ -n "${VIDEO_PATH:-}" ]]; then
  VIDEO_ARGS=(--video-path "${VIDEO_PATH}")
fi

python examples/offline_inference/minicpmo4_5/end2end.py \
  --model-path "${MODEL}" \
  --stage-configs-path "${STAGE_CONFIG}" \
  --query-type use_video \
  --modalities audio \
  --num-video-frames "${NUM_FRAMES}" \
  --output-wav "${OUTPUT}" \
  "${VIDEO_ARGS[@]}"
