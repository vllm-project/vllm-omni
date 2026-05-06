#!/bin/bash
# Marey 30B (Flux-30B-control-v2) multi-keyframe image-to-video curl example,
# using the async video job API.
#
# Reproducer derived from a real moonvalley_ai production payload (see
# params.json in this directory). Two conditioning keyframes (frame index 0
# and 127) constrain a 128-frame 1920x1080 generation; the model fills in
# the in-between motion to interpolate between the two stills.
#
# Server config: pair with run_server.sh (same script as t2v). The 30B's
# default FLOW_SHIFT=3.0 is correct here; flow_shift is also re-asserted
# per request for safety.
#
# Prompt + negative prompt + cond images are loaded from sibling files in
# this directory:
#   prompt.txt
#   negative_prompt.txt
#   frame_0.webp     (target frame index 0)
#   frame_127.webp   (target frame index 127)
#
# Override either prompt by setting PROMPT_FILE / NEGATIVE_PROMPT_FILE in
# the environment, or pre-set PROMPT / NEGATIVE_PROMPT to skip the file load.
# The negative prompt mirrors DEFAULT_NEGATIVE_PROMPT in
# vllm_omni/diffusion/models/marey/pipeline_marey.py.

set -euo pipefail

EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8098}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
SEED="${SEED:-1997074405}"
OUTPUT_PATH="${OUTPUT_PATH:-${EX_DIR}/marey_i2v_multikeyframe_seed${SEED}.mp4}"

PROMPT_FILE="${PROMPT_FILE:-${EX_DIR}/prompt.txt}"
NEGATIVE_PROMPT_FILE="${NEGATIVE_PROMPT_FILE:-${EX_DIR}/negative_prompt.txt}"
PROMPT="${PROMPT:-$(cat "${PROMPT_FILE}")}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-$(cat "${NEGATIVE_PROMPT_FILE}")}"

COND_0="${EX_DIR}/frame_0.webp"
COND_127="${EX_DIR}/frame_127.webp"
for f in "${COND_0}" "${COND_127}"; do
    [[ -f "${f}" ]] || { echo "Missing cond image: ${f}" >&2; exit 1; }
done

# Build frame_conditions JSON dict (OpenAI chat-completions image_url schema)
# pointing at the local cond images via file:// URIs. The server reads them
# from its own filesystem, so this only works when client and server share
# the same machine / mount.
FRAME_CONDITIONS_JSON="$(python3 -c "
import json, os
from urllib.request import pathname2url
out = {}
for idx, p in [('0', '${COND_0}'), ('127', '${COND_127}')]:
    out[idx] = {'image_url': {'url': 'file://' + pathname2url(os.path.abspath(p)), 'detail': 'auto'}}
print(json.dumps(out))
")"

echo "Submitting Marey I2V multi-keyframe request:"
echo "  base_url:    ${BASE_URL}"
echo "  seed:        ${SEED}"
echo "  output_path: ${OUTPUT_PATH}"

create_response="$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    --form-string "prompt=${PROMPT}" \
    --form-string "negative_prompt=${NEGATIVE_PROMPT}" \
    -F "size=1920x1080" \
    -F "num_frames=128" \
    -F "num_inference_steps=33" \
    -F "guidance_scale=4.5" \
    -F "fps=24" \
    -F "flow_shift=3.0" \
    -F "seed=${SEED}" \
    --form-string "frame_conditions=${FRAME_CONDITIONS_JSON}"
)"
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
