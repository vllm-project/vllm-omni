#!/bin/bash

set -euo pipefail

if [ -n "${INPUT_IMAGE:-}" ] && [ ! -f "${INPUT_IMAGE}" ]; then
    echo "INPUT_IMAGE does not exist: ${INPUT_IMAGE}"
    exit 1
fi

BASE_URL="${BASE_URL:-http://localhost:8099}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if [ -n "${INPUT_IMAGE:-}" ]; then
    default_output="lingbot_ti2v.mp4"
else
    default_output="lingbot_t2v.mp4"
fi
OUTPUT_PATH="${OUTPUT_PATH:-${default_output}}"

curl_args=(
    -sS -X POST "${BASE_URL}/v1/videos"
    -H "Accept: application/json"
    -F "prompt=the subject turns toward the camera with smooth natural motion"
    -F "num_frames=9"
    -F "fps=24"
    -F "num_inference_steps=2"
    -F "guidance_scale=3.0"
    -F "flow_shift=3.0"
    -F "seed=42"
)
if [ -n "${INPUT_IMAGE:-}" ]; then
    curl_args+=(
        -F "input_reference=@${INPUT_IMAGE}"
        -F 'extra_params={"size":"320x192"}'
    )
else
    curl_args+=(-F "width=320" -F "height=192")
fi

create_response="$(curl "${curl_args[@]}")"

video_id="$(echo "${create_response}" | jq -er '.id')"
echo "Created video job ${video_id}"

while true; do
    status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
    status="$(echo "${status_response}" | jq -er '.status')"
    case "${status}" in
        queued|in_progress)
            sleep "${POLL_INTERVAL}"
            ;;
        completed)
            break
            ;;
        failed)
            echo "${status_response}" | jq .
            exit 1
            ;;
        *)
            echo "Unexpected video job status: ${status}"
            exit 1
            ;;
    esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved video to ${OUTPUT_PATH}"
