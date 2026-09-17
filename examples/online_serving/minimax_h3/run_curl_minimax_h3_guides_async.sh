#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# MiniMax H3 Ref2VA with timeline guides - async job API with polling
# GUIDE-01 reference case: one ordinary reference image (Picture 1) plus three
# timeline guides placed at frames 36, 72, 120.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8099}"
OUTPUT_PATH="${OUTPUT_PATH:-minimax_h3_guides_output.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"
INPUT_DIR="${INPUT_DIR:-./input}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# The overall_soundscape / non_diegetic_music sections drive H3 audio
# generation, so dropping them changes the audio substantially.
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/multiframe_reference_prompt.txt}"

for file in h3_frame_ref_1.png h3_frame_ref_2.png h3_frame_ref_3.png h3_frame_ref_4.png; do
  if [ ! -f "${INPUT_DIR}/${file}" ]; then
    echo "Error: ${INPUT_DIR}/${file} not found"
    echo "Download the official input images first:"
    echo "  BASE=https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input"
    echo "  wget -P ${INPUT_DIR}/ \${BASE}/h3_frame_ref_{1,2,3,4}.png"
    exit 1
  fi
done

if [ ! -f "${PROMPT_FILE}" ]; then
  echo "Error: prompt file ${PROMPT_FILE} not found"
  exit 1
fi
PROMPT="$(cat "${PROMPT_FILE}")"

# Ordered manifest addressing the guide_files uploads by zero-based index.
TIMELINE_GUIDES='[
  {"frame_index": 36, "image": {"upload_index": 0}},
  {"frame_index": 72, "image": {"upload_index": 1}},
  {"frame_index": 120, "image": {"upload_index": 2}}
]'

# H3 reads task and both sigma shifts from extra_args, not from top-level form
# fields: pipeline_minimax_h3.py resolves extra.get("flow_shift") and
# extra.get("audio_flow_shift"). There is no top-level task or audio_flow_shift
# form field at all, so sending them with -F would be silently ignored.
EXTRA_PARAMS='{"task":"ref2va","aspect_ratio":"16:9","flow_shift":12.0,"audio_flow_shift":3.0}'

echo "Creating MiniMax H3 Ref2VA + timeline guides video job..."
echo "  Prompt file: ${PROMPT_FILE}"
echo "  extra_params: ${EXTRA_PARAMS}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "prompt=${PROMPT}" \
    -F "width=864" \
    -F "height=480" \
    -F "num_frames=124" \
    -F "fps=24" \
    -F "num_inference_steps=20" \
    -F "quality=lossless" \
    -F "seed=148096032077131" \
    -F "extra_params=${EXTRA_PARAMS}" \
    -F "input_references=@${INPUT_DIR}/h3_frame_ref_1.png" \
    -F "timeline_guides=${TIMELINE_GUIDES}" \
    -F "guide_files=@${INPUT_DIR}/h3_frame_ref_2.png" \
    -F "guide_files=@${INPUT_DIR}/h3_frame_ref_3.png" \
    -F "guide_files=@${INPUT_DIR}/h3_frame_ref_4.png"
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
      echo "[$(date +%H:%M:%S)] Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "Video generation completed!"
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

# Download to a temporary file so an HTTP error body is never renamed into a
# .mp4 that looks like a successful result.
TMP_OUTPUT="$(mktemp "${OUTPUT_PATH}.XXXXXX")"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

if ! curl --fail-with-body --silent --show-error -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${TMP_OUTPUT}"; then
  echo "Failed to download video content for job ${video_id}:"
  cat "${TMP_OUTPUT}"
  echo ""
  exit 1
fi

mv "${TMP_OUTPUT}" "${OUTPUT_PATH}"
trap - EXIT
echo "Saved video to ${OUTPUT_PATH}"
