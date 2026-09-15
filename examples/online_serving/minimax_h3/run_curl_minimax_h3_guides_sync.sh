#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# MiniMax H3 Ref2VA with timeline guides - sync endpoint (blocks until complete)
# Uses /v1/videos/sync for one-shot latency measurement without async job storage.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8099}"
OUTPUT_PATH="${OUTPUT_PATH:-minimax_h3_guides_sync_output.mp4}"
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

TIMELINE_GUIDES='[
  {"frame_index": 36, "image": {"upload_index": 0}},
  {"frame_index": 72, "image": {"upload_index": 1}},
  {"frame_index": 120, "image": {"upload_index": 2}}
]'

# H3 reads task and both sigma shifts from extra_args, not from top-level form
# fields. There is no top-level task or audio_flow_shift form field at all.
EXTRA_PARAMS='{"task":"ref2va","aspect_ratio":"16:9","flow_shift":12.0,"audio_flow_shift":3.0}'

echo "Generating MiniMax H3 Ref2VA + timeline guides (sync)..."
echo "  Prompt file: ${PROMPT_FILE}"
echo "  extra_params: ${EXTRA_PARAMS}"
echo "This will block until generation completes."

START_TIME=$(date +%s)

curl -X POST "${BASE_URL}/v1/videos/sync" \
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
  -F "guide_files=@${INPUT_DIR}/h3_frame_ref_4.png" \
  -o "${OUTPUT_PATH}" \
  -w "\nHTTP Status: %{http_code}\nInference Time: %header{X-Inference-Time-S}s\nRequest ID: %header{X-Request-Id}\n"

END_TIME=$(date +%s)
WALL_CLOCK=$((END_TIME - START_TIME))

echo ""
echo "Saved video to ${OUTPUT_PATH}"
echo "Wall-clock time: ${WALL_CLOCK}s"
