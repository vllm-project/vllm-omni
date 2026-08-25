#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Validate MiniMax-H3 FlashGen native LoRA on Ascend NPU.
# Run on a machine with MiniMax-H3 base weights, vllm-omni, and NPU drivers.

set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the MiniMax-H3 base directory}"
: "${FLASHGEN_LORA:?Set FLASHGEN_LORA to minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors}"

API_URL="${API_URL:-http://127.0.0.1:8000/v1/images/generations}"
PROMPT="${PROMPT:-A cinematic sunrise over a calm ocean, gentle waves, stereo ambience.}"
SEED="${SEED:-1101}"

run_case() {
  local label="$1"
  local lora_json="$2"
  local out="/tmp/h3_native_${label}.mp4"
  curl -sS -X POST "${API_URL}" \
    -F "prompt=${PROMPT}" \
    -F "seed=${SEED}" \
    -F 'num_inference_steps=4' \
    -F 'aspect_ratio=16:9' \
    -F 'extra_params={"task":"t2va","duration":5.2}' \
    -F "lora=${lora_json}" \
    -o "${out}"
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "${out}"
  sha256sum "${out}"
}

echo "Base control (scale 0)"
run_case base '{"name":"h3-flashgen","path":"'"${FLASHGEN_LORA}"'","scale":0.0}'

echo "FlashGen native LoRA active"
run_case lora '{"name":"h3-flashgen","path":"'"${FLASHGEN_LORA}"'","scale":1.0}'

echo "Repeat base and LoRA to verify determinism"
run_case base2 '{"name":"h3-flashgen","path":"'"${FLASHGEN_LORA}"'","scale":0.0}'
run_case lora2 '{"name":"h3-flashgen","path":"'"${FLASHGEN_LORA}"'","scale":1.0}'
