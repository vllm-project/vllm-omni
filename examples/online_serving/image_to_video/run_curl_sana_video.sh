#!/bin/bash
# Submit and download a standard SANA-Video-2B image-to-video request.

set -euo pipefail

INPUT_IMAGE="${INPUT_IMAGE:-}"
BASE_URL="${BASE_URL:-http://localhost:8099}"
OUTPUT_PATH="${OUTPUT_PATH:-sana_video_i2v.mp4}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if [ -z "$INPUT_IMAGE" ] || [ ! -f "$INPUT_IMAGE" ]; then
    echo "Set INPUT_IMAGE to an existing image file."
    exit 1
fi

create_response="$(
    curl -sS -X POST "${BASE_URL}/v1/videos" \
        -H "Accept: application/json" \
        -F "prompt=A cat turns toward the camera with smooth, natural motion." \
        -F "negative_prompt=blurry, low quality, temporal artifacts" \
        -F "input_reference=@${INPUT_IMAGE}" \
        -F "width=${WIDTH}" \
        -F "height=${HEIGHT}" \
        -F "num_frames=81" \
        -F "fps=16" \
        -F "num_inference_steps=50" \
        -F "guidance_scale=6.0" \
        -F "seed=42" \
        -F 'extra_params={"motion_score":30}'
)"

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
    echo "Failed to create SANA-Video job:"
    echo "${create_response}" | jq .
    exit 1
fi

while true; do
    status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
    status="$(echo "${status_response}" | jq -r '.status')"

    case "${status}" in
        queued|in_progress)
            echo "Video job ${video_id} status: ${status}"
            sleep "${POLL_INTERVAL}"
            ;;
        completed)
            break
            ;;
        failed)
            echo "SANA-Video generation failed:"
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
echo "Saved SANA-Video to ${OUTPUT_PATH}"
