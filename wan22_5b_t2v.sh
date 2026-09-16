#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/" && pwd)"

# MODEL_ID accepts either a HuggingFace repo id or a local Diffusers model dir.
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-low quality, blurry}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
GUIDANCE_SCALE_HIGH="${GUIDANCE_SCALE_HIGH:-6.0}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-12.0}"
FPS="${FPS:-16}"
GPU_COUNT="${GPU_COUNT:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/wan22_t2v_14b_parallel_bench}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/offline_inference/image_to_video/wan22_i2v_pp2.sh [single|pp|sp|pipefusion]

Parallel modes:
  single      Run without parallelism on the first visible GPU.
  pp          Pipeline parallelism.
  sp          Ulysses sequence parallelism.
  pipefusion  Pipeline parallelism with PipeFusion enabled.

Environment overrides:
  GPU_COUNT=4                  Default card count used to build CUDA_VISIBLE_DEVICES.
  CUDA_VISIBLE_DEVICES=0,1,2,3  Explicit device list.
  MODEL_ID=/path/to/model       Local model dir or HuggingFace repo id.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_CASE="${PARALLEL_MODE:-${1:-pipefusion}}"
case "${RUN_CASE}" in
  single | pp | sp | pipefusion)
    ;;
  *)
    echo "Unknown parallel mode: ${RUN_CASE}" >&2
    usage >&2
    exit 1
    ;;
esac

PROMPTS=(
  "${PROMPT_1:-Cherry blossoms swaying gently in the breeze, petals falling, smooth motion.}"
  "${PROMPT_2:-A red sports car driving along a rainy neon city street at night, cinematic reflections.}"
  "${PROMPT_3:-A golden retriever running through shallow ocean waves at sunset, joyful smooth motion.}"
)

SEEDS=(
  "${SEED_1:-42}"
  "${SEED_2:-43}"
  "${SEED_3:-44}"
)

cd "${REPO_ROOT}"

default_cuda_visible_devices() {
  local count="$1"
  local devices=()

  for ((idx = 0; idx < count; idx++)); do
    devices+=("${idx}")
  done

  local joined="${devices[*]}"
  echo "${joined// /,}"
}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(default_cuda_visible_devices "${GPU_COUNT}")}"

count_devices() {
  local devices="$1"
  local -a parts=()
  IFS=',' read -r -a parts <<< "${devices}"
  echo "${#parts[@]}"
}

NUM_DEVICES="$(count_devices "${CUDA_VISIBLE_DEVICES}")"
if [[ "${RUN_CASE}" == "single" ]]; then
  NUM_DEVICES=1
fi

first_device() {
  local devices="$1"
  echo "${devices%%,*}"
}

filename_slug() {
  local value="$1"
  local slug
  slug="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_//; s/_$//; s/__*/_/g' | cut -c1-80)"
  echo "${slug:-prompt}"
}

run_case() {
  local case_name="$1"
  local width="$2"
  local height="$3"
  local num_frames="$4"
  local prompt="$5"
  local seed="$6"
  local output="$7"
  local log_file="$8"
  local devices="${CUDA_VISIBLE_DEVICES}"
  local num_devices
  local -a env_args=()
  local -a parallel_args=()

  num_devices="$(count_devices "${devices}")"

  case "${case_name}" in
    single)
      devices="${SINGLE_GPU_DEVICE:-$(first_device "${CUDA_VISIBLE_DEVICES}")}"
      parallel_args=()
      ;;
    sp)
      parallel_args=(
        --ulysses-degree "${ULYSSES_DEGREE:-${num_devices}}"
        --vae-patch-parallel-size "${num_devices}"
      )
      ;;
    pipefusion)
      parallel_args=(
        --pipeline-parallel-size "${PIPEFUSION_PIPELINE_PARALLEL_SIZE:-${num_devices}}"
        --vae-patch-parallel-size "${num_devices}"
        --enable-pipefusion
        --pipefusion-warmup-steps "${PIPEFUSION_WARMUP_STEPS:-2}"
        --pipefusion-split-dim "${PIPEFUSION_SPLIT_DIM:-temporal}"
      )
      ;;
    pp)
      parallel_args=(
        --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE:-${num_devices}}"
        --vae-patch-parallel-size "${num_devices}"
      )
      ;;
    *)
      echo "Unknown parallel mode: ${case_name}" >&2
      return 1
      ;;
  esac

  {
    echo "Case: ${case_name}"
    echo "Resolution: ${width}x${height}"
    echo "Frames: ${num_frames}"
    echo "CUDA_VISIBLE_DEVICES: ${devices}"
    echo "Output: ${output}"
    echo "Seed: ${seed}"
    echo "Parallel args: ${parallel_args[*]:-<none>}"
  } | tee "${log_file}"

  env "${env_args[@]}" CUDA_VISIBLE_DEVICES="${devices}" \
    python examples/offline_inference/text_to_video/text_to_video.py \
      --model="${MODEL_ID}" \
      --width="${width}" \
      --height="${height}" \
      --num-frames "${num_frames}" \
      --guidance-scale="${GUIDANCE_SCALE}" \
      --guidance-scale-high="${GUIDANCE_SCALE_HIGH}" \
      --boundary-ratio="${BOUNDARY_RATIO}" \
      --flow-shift="${FLOW_SHIFT}" \
      --fps "${FPS}" \
      --prompt="${prompt}" \
      --negative-prompt="${NEGATIVE_PROMPT}" \
      --output="${output}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS}" \
      --seed "${seed}" \
      --enable-diffusion-pipeline-profiler \
      --enable-cpu-offload \
      --vae-use-tiling \
      "${parallel_args[@]}" 2>&1 | tee -a "${log_file}"
}

GIT_COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"
echo "Git commit SHA: ${GIT_COMMIT_SHA}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Parallel mode: ${RUN_CASE}"
echo "Default GPU count: ${GPU_COUNT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Number of devices: ${NUM_DEVICES}"
mkdir -p "${OUTPUT_DIR}"

CONFIGS=(
  "832 480 81"
  "832 480 121"
  "1280 704 81"
  "1280 704 121"
)

for config in "${CONFIGS[@]}"; do
  read -r width height num_frames <<< "${config}"
  run_dir="${OUTPUT_DIR}/${width}x${height}x${num_frames}/${RUN_CASE}_${NUM_DEVICES}devices"
  mkdir -p "${run_dir}"

  for idx in "${!PROMPTS[@]}"; do
    run_id=$((idx + 1))
    prompt_slug="$(filename_slug "${PROMPTS[$idx]}")"
    file_stem="${width}x${height}x${num_frames}_${RUN_CASE}_run_${run_id}_prompt_${prompt_slug}_seed_${SEEDS[$idx]}"
    output="${run_dir}/${file_stem}.mp4"
    log_file="${run_dir}/${file_stem}.log"

    echo "Starting ${RUN_CASE} ${width}x${height}x${num_frames} run ${run_id}/${#PROMPTS[@]}: output=${output}"
    if run_case "${RUN_CASE}" "${width}" "${height}" "${num_frames}" "${PROMPTS[$idx]}" "${SEEDS[$idx]}" "${output}" "${log_file}"; then
      echo "Finished ${RUN_CASE} ${width}x${height}x${num_frames} run ${run_id}/${#PROMPTS[@]}"
    else
      status=$?
      echo "Run failed with exit code ${status}; continuing to the next test." | tee -a "${log_file}"
    fi
  done
done
