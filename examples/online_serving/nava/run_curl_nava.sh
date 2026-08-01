#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-nava}"
OUTPUT_PATH="${OUTPUT_PATH:-nava_output.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

response="$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "model=${MODEL}" \
    -F "prompt=A person speaks while standing near the sea." \
    -F 'extra_params={"num_frames":37,"fps":24,"num_inference_steps":50}'
)"

if ! video_id="$(echo "${response}" | jq -r '.id')"; then
  echo "Failed to parse video job creation response:"
  echo "${response}"
  exit 1
fi

if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
echo "${response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  if ! status="$(echo "${status_response}" | jq -r '.status')"; then
    echo "Failed to parse video job status response:"
    echo "${status_response}"
    exit 1
  fi

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
