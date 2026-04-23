#!/bin/bash
# OmniWeaving image-to-video curl example using the async video job API.

set -euo pipefail

INPUT_IMAGE="${INPUT_IMAGE:-test_input.jpg}"
BASE_URL="${BASE_URL:-http://localhost:8096}"
OUTPUT_PATH="${OUTPUT_PATH:-omniweaving_i2v.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    echo "Provide an image via INPUT_IMAGE env var."
    exit 1
fi

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "prompt=The camera follows the puppy as it runs forward on the grass." \
    -F "input_reference=@${INPUT_IMAGE}" \
    -F "size=832x480" \
    -F "num_frames=33" \
    -F "fps=8" \
    -F "num_inference_steps=30" \
    -F "guidance_scale=6.0" \
    -F "flow_shift=5.0" \
    -F "seed=42"
)

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
echo "${create_response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq .
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved video to ${OUTPUT_PATH}"
