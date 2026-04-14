#!/bin/bash
# Wan2.2 image-to-video curl example using the async video job API.

set -euo pipefail

INPUT_IMAGE="${INPUT_IMAGE:-../../offline_inference/image_to_video/qwen-bear.png}"
BASE_URL="${BASE_URL:-http://localhost:8099}"
OUTPUT_PATH="${OUTPUT_PATH:-wan22_i2v_output.mp4}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    exit 1
fi

create_cmd=(
  curl -sS -X POST "${BASE_URL}/v1/videos"
  -H "Accept: application/json"
  -F "prompt=A bear playing with yarn, smooth motion"
  -F "input_reference=@${INPUT_IMAGE}"
  -F "seconds=2"
  -F "size=832x480"
  -F "fps=16"
  -F "num_inference_steps=40"
  -F "guidance_scale=1.0"
  -F "guidance_scale_2=1.0"
  -F "boundary_ratio=0.875"
  -F "flow_shift=12.0"
  -F "seed=42"
)

if [ -n "${NEGATIVE_PROMPT}" ]; then
  create_cmd+=(-F "negative_prompt=${NEGATIVE_PROMPT}")
fi

if [ -n "${SAMPLE_SOLVER}" ]; then
  create_cmd+=(-F "extra_params={\"sample_solver\":\"${SAMPLE_SOLVER}\"}")
fi

create_response="$("${create_cmd[@]}")"
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
