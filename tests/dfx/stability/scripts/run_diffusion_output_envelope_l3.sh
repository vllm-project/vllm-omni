#!/usr/bin/env bash
set -euo pipefail

# L3-style smoke for the diffusion payload/metadata output path.
#
# Usage:
#   bash tests/dfx/stability/scripts/run_diffusion_output_envelope_l3.sh quick
#   bash tests/dfx/stability/scripts/run_diffusion_output_envelope_l3.sh full
#
# Optional knobs:
#   VENV=/mnt/data4/bjf_workspace/.venv_rdma
#   HF_HOME=/mnt/data1/huggingface
#   OUT_DIR=/tmp/diffusion_output_envelope_l3
#   RUN_PROTOCOL_TESTS=1
#   RUN_QWEN_IMAGE=1
#   RUN_COSMOS_VIDEO=1
#   RUN_COSMOS_ACTION=1
#   RUN_WAN_I2V=1
#   EXTRA_MODEL_ARGS="--enforce-eager"

MODE="${1:-quick}"
if [[ "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [quick|full]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VENV="${VENV:-/mnt/data4/bjf_workspace/.venv_rdma}"
PYTHON="${VENV}/bin/python"
HF_HOME="${HF_HOME:-/mnt/data1/huggingface}"
OUT_DIR="${OUT_DIR:-/tmp/diffusion_output_envelope_l3}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:-${HF_HOME}/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6}"
COSMOS3_MODEL="${COSMOS3_MODEL:-${HF_HOME}/hub/models--nvidia--Cosmos3-Nano/snapshots/23314034b82f46b45339035dba67c3ee9bbcb8ba}"
WAN22_I2V_MODEL="${WAN22_I2V_MODEL:-${HF_HOME}/hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/snapshots/596658fd9ca6b7b71d5057529bbf319ecbc61d74}"
WAN22_I2V_IMAGE="${WAN22_I2V_IMAGE:-${WAN22_I2V_MODEL}/examples/i2v_input.JPG}"

RUN_PROTOCOL_TESTS="${RUN_PROTOCOL_TESTS:-1}"
RUN_QWEN_IMAGE="${RUN_QWEN_IMAGE:-1}"
RUN_COSMOS_VIDEO="${RUN_COSMOS_VIDEO:-1}"
RUN_COSMOS_ACTION="${RUN_COSMOS_ACTION:-1}"
RUN_WAN_I2V="${RUN_WAN_I2V:-1}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"

mkdir -p "${OUT_DIR}/logs"

run_case() {
  local name="$1"
  shift
  local log_file="${OUT_DIR}/logs/${name}.log"
  echo
  echo "================ ${name} ================"
  echo "log: ${log_file}"
  local start
  start="$(date +%s)"
  "$@" 2>&1 | tee "${log_file}"
  local end
  end="$(date +%s)"
  echo "[${name}] elapsed=$((end - start))s"
}

require_file() {
  local file="$1"
  if [[ ! -s "${file}" ]]; then
    echo "Expected non-empty output file not found: ${file}" >&2
    exit 1
  fi
}

echo "mode=${MODE}"
echo "root=${ROOT_DIR}"
echo "python=${PYTHON}"
echo "hf_home=${HF_HOME}"
echo "out_dir=${OUT_DIR}"

cd "${ROOT_DIR}"

if [[ "${RUN_PROTOCOL_TESTS}" == "1" ]]; then
  run_case protocol_formatter_tests \
    "${PYTHON}" -m pytest \
      tests/diffusion/test_diffusion_output_metadata.py \
      tests/diffusion/test_diffusion_output_formatter.py
fi

if [[ "${MODE}" == "quick" ]]; then
  QWEN_STEPS=4
  QWEN_HEIGHT=1024
  QWEN_WIDTH=1024
  COSMOS_STEPS=4
  COSMOS_FRAMES=17
  COSMOS_HEIGHT=256
  COSMOS_WIDTH=320
  WAN_STEPS=4
  WAN_FRAMES=17
  WAN_HEIGHT=256
  WAN_WIDTH=448
else
  QWEN_STEPS=50
  QWEN_HEIGHT=1024
  QWEN_WIDTH=1024
  COSMOS_STEPS=35
  COSMOS_FRAMES=189
  COSMOS_HEIGHT=720
  COSMOS_WIDTH=1280
  WAN_STEPS=50
  WAN_FRAMES=81
  WAN_HEIGHT=480
  WAN_WIDTH=832
fi

if [[ "${RUN_QWEN_IMAGE}" == "1" ]]; then
  QWEN_OUT="${OUT_DIR}/qwen_image_t2i.png"
  run_case qwen_image_t2i \
    "${PYTHON}" examples/offline_inference/text_to_image/text_to_image.py \
      --model "${QWEN_IMAGE_MODEL}" \
      --prompt "A clean studio photo of a red cube on a white table." \
      --height "${QWEN_HEIGHT}" \
      --width "${QWEN_WIDTH}" \
      --num-inference-steps "${QWEN_STEPS}" \
      --seed 11 \
      --output "${QWEN_OUT}" \
      ${EXTRA_MODEL_ARGS}
  require_file "${QWEN_OUT}"
fi

COSMOS_COMMON_EXTRA='{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false, "use_resolution_template": false, "use_duration_template": false}'

if [[ "${RUN_COSMOS_VIDEO}" == "1" ]]; then
  COSMOS_VIDEO_OUT="${OUT_DIR}/cosmos3_t2v.mp4"
  run_case cosmos3_t2v \
    "${PYTHON}" examples/offline_inference/text_to_video/text_to_video.py \
      --model "${COSMOS3_MODEL}" \
      --prompt "A robot arm moves smoothly over a workbench." \
      --negative-prompt "blurry, distorted, low quality" \
      --height "${COSMOS_HEIGHT}" \
      --width "${COSMOS_WIDTH}" \
      --num-frames "${COSMOS_FRAMES}" \
      --fps 24 \
      --num-inference-steps "${COSMOS_STEPS}" \
      --guidance-scale 6.0 \
      --extra-body "${COSMOS_COMMON_EXTRA}" \
      --output "${COSMOS_VIDEO_OUT}" \
      ${EXTRA_MODEL_ARGS}
  require_file "${COSMOS_VIDEO_OUT}"
fi

if [[ "${RUN_COSMOS_ACTION}" == "1" ]]; then
  COSMOS_ACTION_OUT="${OUT_DIR}/cosmos3_forward_dynamics.mp4"
  COSMOS_ACTION_EXTRA='{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false, "use_resolution_template": false, "use_duration_template": false, "action_mode": "forward_dynamics", "action": [[0.1, 0.2], [0.3, 0.4]], "raw_action_dim": 2, "domain_name": "av", "action_chunk_size": 2}'
  run_case cosmos3_forward_dynamics_action \
    "${PYTHON}" examples/offline_inference/text_to_video/text_to_video.py \
      --model "${COSMOS3_MODEL}" \
      --prompt "Forward dynamics action smoke test." \
      --height "${COSMOS_HEIGHT}" \
      --width "${COSMOS_WIDTH}" \
      --num-frames 3 \
      --fps 4 \
      --num-inference-steps "${COSMOS_STEPS}" \
      --guidance-scale 3.0 \
      --extra-body "${COSMOS_ACTION_EXTRA}" \
      --output "${COSMOS_ACTION_OUT}" \
      ${EXTRA_MODEL_ARGS}
  require_file "${COSMOS_ACTION_OUT}"
fi

if [[ "${RUN_WAN_I2V}" == "1" ]]; then
  WAN_OUT="${OUT_DIR}/wan22_i2v.mp4"
  run_case wan22_i2v \
    "${PYTHON}" examples/offline_inference/image_to_video/image_to_video.py \
      --model "${WAN22_I2V_MODEL}" \
      --image "${WAN22_I2V_IMAGE}" \
      --prompt "The scene comes to life with smooth camera motion." \
      --height "${WAN_HEIGHT}" \
      --width "${WAN_WIDTH}" \
      --num-frames "${WAN_FRAMES}" \
      --num-inference-steps "${WAN_STEPS}" \
      --guidance-scale 5.0 \
      --fps 16 \
      --vae-use-tiling \
      --output "${WAN_OUT}" \
      ${EXTRA_MODEL_ARGS}
  require_file "${WAN_OUT}"
fi

echo
echo "All selected diffusion output envelope L3 smoke cases passed."
echo "Outputs: ${OUT_DIR}"
