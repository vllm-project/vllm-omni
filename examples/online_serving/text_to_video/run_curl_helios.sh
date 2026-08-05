#!/bin/bash
# Helios text-to-video curl example using the async video job API.
#
# Helios-specific knobs (declared in vllm_omni/model_extras/helios.py) are passed
# through the generic `extra_params` JSON form field. Select a variant via PRESET:
#   PRESET=base        -> Helios-Base, Stage 1 only (default)
#   PRESET=mid-stage2  -> Helios-Mid, Stage 2 pyramid + CFG-Zero*
#   PRESET=distilled   -> Helios-Distilled, Stage 2 pyramid + DMD (few-step)

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8098}"
MODEL="${MODEL:-BestWishYsh/Helios-Base}"
PROMPT="${PROMPT:-A dynamic time-lapse of scenery rushing past the window of a speeding train.}"
PRESET="${PRESET:-base}"
OUTPUT_PATH="${OUTPUT_PATH:-helios_t2v_${PRESET}.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

case "${PRESET}" in
  base)
    EXTRA_PARAMS=""
    GUIDANCE_SCALE="5.0"
    ;;
  mid-stage2)
    EXTRA_PARAMS='{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [20, 20, 20], "use_cfg_zero_star": true, "use_zero_init": true, "zero_steps": 1}'
    GUIDANCE_SCALE="5.0"
    ;;
  distilled)
    EXTRA_PARAMS='{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [2, 2, 2], "is_amplify_first_chunk": true}'
    GUIDANCE_SCALE="1.0"
    ;;
  *)
    echo "Unknown PRESET '${PRESET}' (expected base|mid-stage2|distilled)"
    exit 1
    ;;
esac

create_args=(
  -sS -X POST "${BASE_URL}/v1/videos"
  -H "Accept: application/json"
  -F "prompt=${PROMPT}"
  -F "model=${MODEL}"
  -F "size=640x384"
  -F "num_frames=99"
  -F "fps=16"
  -F "num_inference_steps=50"
  -F "guidance_scale=${GUIDANCE_SCALE}"
  -F "seed=42"
)
if [ -n "${EXTRA_PARAMS}" ]; then
  create_args+=(-F "extra_params=${EXTRA_PARAMS}")
fi

create_response=$(curl "${create_args[@]}")

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id} (preset=${PRESET})"
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
