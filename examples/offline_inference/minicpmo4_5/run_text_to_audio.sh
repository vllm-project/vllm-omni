#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/minicpmo4_5.yaml}"
OUTPUT="${OUTPUT:-minicpmo45_text_to_audio.wav}"
PROMPT="${PROMPT:-Please read this sentence aloud: vLLM Omni is testing MiniCPM text to audio generation.}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-When audio output is requested, reply with speech only and follow any requested length constraints.}"

python examples/offline_inference/minicpmo4_5/end2end.py \
  --model-path "${MODEL}" \
  --deploy-config "${DEPLOY_CONFIG}" \
  --query-type text \
  --modalities audio \
  --prompt "${PROMPT}" \
  --system-prompt "${SYSTEM_PROMPT}" \
  --output-wav "${OUTPUT}"
