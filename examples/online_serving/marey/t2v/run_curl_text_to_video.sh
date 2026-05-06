#!/bin/bash
# Marey 30B (Flux-30B-control-v2) text-to-video curl example using the async
# video job API.
#
# Matches the hparams from test.sh (offline inference):
#   --height 1080 --width 1920 --num-frames 128 --steps 33 --guidance-scale 3.5
#
# Server config: any FLOW_SHIFT works — the per-request `flow_shift=3.0`
# below overrides whatever the server was started with.
#
# Prompt and negative prompt are loaded from sibling text files. Override
# either by setting PROMPT_FILE / NEGATIVE_PROMPT_FILE in the environment, or
# pre-set PROMPT / NEGATIVE_PROMPT to skip the file load entirely.
#
# The negative prompt mirrors DEFAULT_NEGATIVE_PROMPT in
# vllm_omni/diffusion/models/marey/pipeline_marey.py and the moonvalley
# `marey_inference.py --negative-prompt` reference; sending it explicitly
# keeps this client self-contained and parity-stable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8098}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
SEED="${SEED:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-marey_output_noquality_seed${SEED}.mp4}"

PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompt.txt}"
NEGATIVE_PROMPT_FILE="${NEGATIVE_PROMPT_FILE:-${SCRIPT_DIR}/negative_prompt.txt}"
PROMPT="${PROMPT:-$(cat "${PROMPT_FILE}")}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$(cat "${NEGATIVE_PROMPT_FILE}")}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    --form-string "prompt=${PROMPT}" \
    --form-string "negative_prompt=${NEGATIVE_PROMPT}" \
    -F "size=1920x1080" \
    -F "num_frames=128" \
    -F "num_inference_steps=33" \
    -F "guidance_scale=3.5" \
    -F "flow_shift=3.0" \
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
