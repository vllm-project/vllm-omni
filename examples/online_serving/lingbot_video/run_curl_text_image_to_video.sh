#!/bin/bash

set -euo pipefail

if [ -z "${INPUT_IMAGE:-}" ] || [ ! -f "${INPUT_IMAGE}" ]; then
    echo "Set INPUT_IMAGE to an existing first-frame image."
    exit 1
fi

BASE_URL="${BASE_URL:-http://localhost:8099}"
MODEL="${MODEL:-robbyant/lingbot-video-dense-1.3b}"
OUTPUT_PATH="${OUTPUT_PATH:-lingbot_ti2v.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

create_response="$(curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "model=${MODEL}" \
    -F "prompt=the subject turns toward the camera with smooth natural motion" \
    -F "input_reference=@${INPUT_IMAGE}" \
    -F "size=320x192" \
    -F "num_frames=9" \
    -F "fps=24" \
    -F "num_inference_steps=2" \
    -F "guidance_scale=3.0" \
    -F "flow_shift=3.0" \
    -F "seed=42")"

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
