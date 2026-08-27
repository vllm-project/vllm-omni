#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Validate MiniMax-H3 FlashGen native LoRA on Ascend NPU.
# Run on a machine with MiniMax-H3 base weights, vllm-omni, and NPU drivers.

set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the MiniMax-H3 base directory}"
: "${FLASHGEN_LORA:?Set FLASHGEN_LORA to minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors}"

API_URL="${API_URL:-http://127.0.0.1:8000/v1/videos/sync}"
PROMPT="${PROMPT:-A cinematic sunrise over a calm ocean, gentle waves, stereo ambience.}"
SEED="${SEED:-1101}"

# Prints only the decode hash on stdout so callers can compare runs; progress
# and stream probes go to stderr.
run_case() {
  local label="$1"
  local scale="$2"
  local out="/tmp/h3_native_${label}.mp4"
  curl -sS --fail -X POST "${API_URL}" \
    -F "prompt=${PROMPT}" \
    -F "seed=${SEED}" \
    -F 'num_inference_steps=4' \
    -F 'aspect_ratio=16:9' \
    -F 'extra_params={"task":"t2va","duration":5.2}' \
    -F "lora={\"name\":\"h3-flashgen\",\"path\":\"${FLASHGEN_LORA}\",\"scale\":${scale}}" \
    -o "${out}"
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "${out}" >&2
  sha256sum "${out}" | cut -d' ' -f1
}

echo "Base control (scale 0)" >&2
base="$(run_case base 0.0)"
echo "FlashGen native LoRA active (scale 1)" >&2
lora="$(run_case lora 1.0)"
echo "Repeat both to verify determinism" >&2
base2="$(run_case base2 0.0)"
lora2="$(run_case lora2 1.0)"

printf 'base =%s\nbase2=%s\nlora =%s\nlora2=%s\n' "${base}" "${base2}" "${lora}" "${lora2}"

status=0
if [[ "${base}" != "${base2}" ]]; then
  echo "FAIL: base decode is nondeterministic across repeats" >&2
  status=1
fi
if [[ "${lora}" != "${lora2}" ]]; then
  echo "FAIL: LoRA decode is nondeterministic across repeats" >&2
  status=1
fi
if [[ "${base}" == "${lora}" ]]; then
  echo "FAIL: LoRA activation did not change the decode stream" >&2
  status=1
fi
if [[ "${status}" -eq 0 ]]; then
  echo "PASS: deterministic per scale, and LoRA activation changes the output" >&2
fi
exit "${status}"
