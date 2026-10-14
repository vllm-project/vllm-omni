#!/bin/bash
# Wan2.2 S2V (speech-to-video) curl example using the sync video API.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-s2v_480p_serve.mp4}"
IMAGE_URL="${IMAGE_URL:-https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.png}"
AUDIO_URL="${AUDIO_URL:-https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.MP3}"
PROMPT="${PROMPT:-A person singing}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
FPS="${FPS:-16}"

echo "Sending S2V request..."
echo "  Image URL: $IMAGE_URL"
echo "  Audio URL: $AUDIO_URL"
echo "  Prompt: $PROMPT"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Steps: $NUM_INFERENCE_STEPS"
echo "  FPS: $FPS"

IMAGE_REF_JSON="{\"image_url\": \"${IMAGE_URL}\"}"
AUDIO_REF_JSON="{\"audio_url\": \"${AUDIO_URL}\"}"

no_proxy=127.0.0.1 \
curl -X POST "${BASE_URL}/v1/videos/sync" \
  -F "prompt=${PROMPT}" \
  -F "image_reference=${IMAGE_REF_JSON}" \
  -F "audio_reference=${AUDIO_REF_JSON}" \
  -F "width=${WIDTH}" -F "height=${HEIGHT}" \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F "guidance_scale=${GUIDANCE_SCALE}" \
  -F "fps=${FPS}" \
  --output "${OUTPUT_PATH}"

if [ -f "$OUTPUT_PATH" ] && [ -s "$OUTPUT_PATH" ]; then
    echo "Saved video to ${OUTPUT_PATH} ($(du -h "$OUTPUT_PATH" | cut -f1))"
else
    echo "ERROR: Output file is empty or missing"
    exit 1
fi
