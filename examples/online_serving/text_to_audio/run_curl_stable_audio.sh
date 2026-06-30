#!/bin/bash
# Stable Audio Open text-to-audio curl example.
#
# Stable Audio uses the OpenAI-compatible `POST /v1/audio/generate` endpoint:
# a JSON request in, binary audio (WAV by default) out. The generated audio is
# written directly to OUTPUT_PATH via curl's --output.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
PROMPT="${PROMPT:-A piano playing a gentle melody}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-Low quality, distorted, noisy}"
AUDIO_LENGTH="${AUDIO_LENGTH:-10.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SEED="${SEED:-42}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-wav}"
OUTPUT_PATH="${OUTPUT_PATH:-stable_audio_output.wav}"

read -r -d '' PAYLOAD <<EOF || true
{
  "input": "${PROMPT}",
  "negative_prompt": "${NEGATIVE_PROMPT}",
  "audio_length": ${AUDIO_LENGTH},
  "guidance_scale": ${GUIDANCE_SCALE},
  "num_inference_steps": ${NUM_INFERENCE_STEPS},
  "seed": ${SEED},
  "response_format": "${RESPONSE_FORMAT}"
}
EOF

echo "POST ${BASE_URL}/v1/audio/generate"
curl -sS -X POST "${BASE_URL}/v1/audio/generate" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  --output "${OUTPUT_PATH}"

echo "Saved generated audio to ${OUTPUT_PATH}"
