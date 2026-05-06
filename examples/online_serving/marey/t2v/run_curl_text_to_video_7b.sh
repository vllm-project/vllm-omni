#!/bin/bash
# Marey 7B (Flux-7B) text-to-video curl example using the async video job API.
#
# The 7B was trained at up to 480p (bucket_config max).
#
# Resolution constraint: H_patches * W_patches must be divisible by
# ULYSSES_DEGREE (8). For 480p the safe choice is 768x480 (patches 24x15,
# S=360, 360/8=45). 848x480 fails (patches 27x15, S=405, not div by 8).
# 1280x720 also works (patches 40x23, S=920/8=115) but is above training range.
#
# The checkpoint is not distilled → needs ~50 steps (vs 33 for distilled 30B)
# and a higher guidance scale (~7.0 vs 3.5).
#
# Server config: any FLOW_SHIFT works — the per-request `flow_shift=0`
# below overrides whatever the server was started with. (7B requires 0;
# the 30B's default 3.0 produces broken outputs.)
#
# Prompt and negative prompt are loaded from sibling text files. Override
# either by setting PROMPT_FILE / NEGATIVE_PROMPT_FILE in the environment, or
# pre-set PROMPT / NEGATIVE_PROMPT to skip the file load entirely. The
# negative prompt mirrors DEFAULT_NEGATIVE_PROMPT in
# vllm_omni/diffusion/models/marey/pipeline_marey.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8098}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
SEED="${SEED:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-marey_7b_output_noquality_seed${SEED}.mp4}"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompt.txt}"
NEGATIVE_PROMPT_FILE="${NEGATIVE_PROMPT_FILE:-${SCRIPT_DIR}/negative_prompt.txt}"
PROMPT="${PROMPT:-$(cat "${PROMPT_FILE}")}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$(cat "${NEGATIVE_PROMPT_FILE}")}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    --form-string "prompt=${PROMPT}" \
    --form-string "negative_prompt=${NEGATIVE_PROMPT}" \
    -F "size=768x480" \
    -F "num_frames=64" \
    -F "num_inference_steps=50" \
    -F "guidance_scale=7.0" \
    -F "flow_shift=0" \
    -F "seed=${SEED}"
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
