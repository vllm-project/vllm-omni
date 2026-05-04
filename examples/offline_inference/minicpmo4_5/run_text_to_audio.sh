#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/minicpmo4_5.yaml}"
OUTPUT="${OUTPUT:-minicpmo45_text_to_audio.wav}"
PROMPT="${PROMPT:-Please read this single long sentence aloud exactly once without shortening it: vLLM Omni is running a benchmark for MiniCPM speech generation, and this sentence intentionally includes enough detail about streaming text to audio generation, multimodal reasoning, stage connectors, careful benchmarking, and stable speech synthesis behavior to last well over ten seconds when spoken at a natural pace.}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-When audio output is requested, reply with speech only and follow any requested length constraints.}"
REF_AUDIO_ARGS=()

if [[ -n "${REF_AUDIO_PATH:-}" ]]; then
  REF_AUDIO_ARGS=(--ref-audio-path "${REF_AUDIO_PATH}")
fi

python examples/offline_inference/minicpmo4_5/end2end.py \
  --model-path "${MODEL}" \
  --deploy-config "${DEPLOY_CONFIG}" \
  --query-type text \
  --modalities audio \
  --prompt "${PROMPT}" \
  --system-prompt "${SYSTEM_PROMPT}" \
  --output-wav "${OUTPUT}" \
  "${REF_AUDIO_ARGS[@]}"
