#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# LTX-2 text-to-video curl example using the async video job API.
# Start the server first: bash run_server_ltx2.sh best-combo

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8098}"
OUTPUT_PATH="${OUTPUT_PATH:-ltx2_output.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

PROMPT="${PROMPT:-A serene lakeside sunrise with mist over the water.}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "prompt=${PROMPT}" \
    -F "width=768" \
    -F "height=480" \
    -F "num_frames=41" \
    -F "fps=24" \
    -F "num_inference_steps=20" \
    -F "guidance_scale=3.0" \
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
