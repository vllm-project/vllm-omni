#!/bin/bash

set -euo pipefail

if [ -n "${INPUT_IMAGE:-}" ] && [ ! -f "${INPUT_IMAGE}" ]; then
    echo "INPUT_IMAGE does not exist: ${INPUT_IMAGE}"
    exit 1
fi

BASE_URL="${BASE_URL:-http://localhost:8099}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
RUN_REFINER="${RUN_REFINER:-false}"
REFINER_HEIGHT="${REFINER_HEIGHT:-192}"
REFINER_WIDTH="${REFINER_WIDTH:-320}"
REFINER_STEPS="${REFINER_STEPS:-2}"
REFINER_GUIDANCE_SCALE="${REFINER_GUIDANCE_SCALE:-3.0}"
REFINER_SHIFT="${REFINER_SHIFT:-3.0}"
REFINER_T_THRESH="${REFINER_T_THRESH:-0.85}"
REFINER_SIGMA_TAIL_STEPS="${REFINER_SIGMA_TAIL_STEPS:-2}"
REFINER_SAMPLE_FPS="${REFINER_SAMPLE_FPS:-24}"
REFINER_OUTPUT_FPS="${REFINER_OUTPUT_FPS:-24}"
REFINER_MAX_VIDEO_FRAMES="${REFINER_MAX_VIDEO_FRAMES:-9}"

refiner_extra=""
if [ "${RUN_REFINER}" = "true" ]; then
    refiner_extra="\"run_refiner\":true,\"refiner_height\":${REFINER_HEIGHT},\"refiner_width\":${REFINER_WIDTH},\"refiner_steps\":${REFINER_STEPS},\"refiner_guidance_scale\":${REFINER_GUIDANCE_SCALE},\"refiner_shift\":${REFINER_SHIFT},\"refiner_t_thresh\":${REFINER_T_THRESH},\"refiner_sigma_tail_steps\":${REFINER_SIGMA_TAIL_STEPS},\"refiner_sample_fps\":${REFINER_SAMPLE_FPS},\"refiner_output_fps\":${REFINER_OUTPUT_FPS},\"refiner_max_video_frames\":${REFINER_MAX_VIDEO_FRAMES}"
fi

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
    extra_params='{"size":"320x192"}'
    if [ "${RUN_REFINER}" = "true" ]; then
        extra_params="{\"size\":\"320x192\",${refiner_extra}}"
    fi
    curl_args+=(
        -F "input_reference=@${INPUT_IMAGE}"
        -F "extra_params=${extra_params}"
    )
else
    curl_args+=(-F "width=320" -F "height=192")
    if [ "${RUN_REFINER}" = "true" ]; then
        curl_args+=(-F "extra_params={${refiner_extra}}")
    fi
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
