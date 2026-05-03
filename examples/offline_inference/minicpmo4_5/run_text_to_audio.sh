#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/minicpmo.yaml}"
OUTPUT="${OUTPUT:-minicpmo45_text_to_audio.wav}"

python examples/offline_inference/minicpmo4_5/end2end.py \
  --model-path "${MODEL}" \
  --stage-configs-path "${STAGE_CONFIG}" \
  --query-type text \
  --modalities audio \
  --output-wav "${OUTPUT}"
