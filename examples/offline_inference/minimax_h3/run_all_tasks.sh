#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# Reproducible MiniMax-H3 multi-GPU runner. Defaults to the recommended four-GPU
# TP2 x Ulysses2 topology and the T2V/I2V (FL2VA) checkpoint partition.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname -- "${REPO_ROOT}")}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/MiniMax-H3}"

NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-2}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-2}"
RING_DEGREE="${RING_DEGREE:-1}"
TEXT_ENCODER_TP_SIZE="${TEXT_ENCODER_TP_SIZE:-4}"
VAE_PATCH_PARALLEL_SIZE="${VAE_PATCH_PARALLEL_SIZE:-4}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUDNN_ATTN}"
ENABLE_DLO="${ENABLE_DLO:-0}"
DLO_RESIDENT_LAYERS="${DLO_RESIDENT_LAYERS:-0}"
RUN_REF2VA="${RUN_REF2VA:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NUM_GPUS - 1))")"
fi

TOPOLOGY="tp${TP_SIZE}-u${ULYSSES_DEGREE}-r${RING_DEGREE}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/results/minimax-h3-${TOPOLOGY}-$(date -u +%Y%m%dT%H%M%SZ)}"
RUNNER="${SCRIPT_DIR}/all_tasks.py"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1344}"
DURATION_SECONDS="${DURATION_SECONDS:-5.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
SEED_BASE="${SEED_BASE:-1101}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
MAX_PREFLIGHT_MEMORY_MIB="${MAX_PREFLIGHT_MEMORY_MIB:-2048}"
MAX_PREFLIGHT_GPU_UTIL="${MAX_PREFLIGHT_GPU_UTIL:-10}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-70000}"
PROFILE_DIR="${PROFILE_DIR:-}"
FP8_Q_SCALE="${FP8_Q_SCALE:-}"
FP8_K_SCALE="${FP8_K_SCALE:-}"
FP8_V_SCALE="${FP8_V_SCALE:-}"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "${WORK_ROOT}/bin/python" ]]; then
  PYTHON="${WORK_ROOT}/bin/python"
elif [[ -x "${WORK_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${WORK_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="$(command -v python || command -v python3 || true)"
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "No Python interpreter found. Set PYTHON=/path/to/venv/bin/python." >&2
  exit 1
fi

REQUIRED_FILES=(
  "${RUNNER}"
  "${MODEL_ROOT}/FL2VA/model_index.json"
)
if [[ "${RUN_REF2VA}" == "1" ]]; then
  REQUIRED_FILES+=("${MODEL_ROOT}/Ref2VA/model_index.json")
fi
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file is missing: ${required_file}" >&2
    exit 1
  fi
done

for command_name in nvidia-smi ffmpeg ffprobe; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "Required command is missing: ${command_name}" >&2
    exit 1
  fi
done

IFS=',' read -r -a SELECTED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#SELECTED_GPUS[@]}" -ne "${NUM_GPUS}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly ${NUM_GPUS} physical GPU indices." >&2
  exit 1
fi
if (( TP_SIZE * ULYSSES_DEGREE * RING_DEGREE != NUM_GPUS )); then
  echo "NUM_GPUS must equal TP_SIZE * ULYSSES_DEGREE * RING_DEGREE." >&2
  exit 1
fi

GPU_STATE="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)"
echo "Selected physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Topology: ${TOPOLOGY}; attention: ${ATTENTION_BACKEND}"
echo "${GPU_STATE}"
for gpu_index in "${SELECTED_GPUS[@]}"; do
  used_mib="$(awk -F',' -v wanted="${gpu_index}" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
      if ($1 == wanted) print $4
    }
  ' <<< "${GPU_STATE}")"
  total_mib="$(awk -F',' -v wanted="${gpu_index}" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
      if ($1 == wanted) print $3
    }
  ' <<< "${GPU_STATE}")"
  gpu_util="$(awk -F',' -v wanted="${gpu_index}" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5)
      if ($1 == wanted) print $5
    }
  ' <<< "${GPU_STATE}")"
  if [[ -z "${used_mib}" ]]; then
    echo "GPU ${gpu_index} was not found by nvidia-smi." >&2
    exit 1
  fi
  if (( total_mib < MIN_GPU_MEMORY_MIB )); then
    echo "GPU ${gpu_index} has ${total_mib} MiB; at least ${MIN_GPU_MEMORY_MIB} MiB is required." >&2
    exit 1
  fi
  if (( used_mib > MAX_PREFLIGHT_MEMORY_MIB )); then
    echo "GPU ${gpu_index} already uses ${used_mib} MiB; refusing to interfere." >&2
    exit 1
  fi
  if (( gpu_util > MAX_PREFLIGHT_GPU_UTIL )); then
    echo "GPU ${gpu_index} is ${gpu_util}% busy; refusing to interfere." >&2
    exit 1
  fi
done

mkdir -p \
  "${OUTPUT_DIR}" \
  "${WORK_ROOT}/hf-cache" \
  "${WORK_ROOT}/torchinductor-cache" \
  "${WORK_ROOT}/triton-cache" \
  "${WORK_ROOT}/xdg-cache/torch/kernels"

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
  "${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${WORK_ROOT}/hf-cache"
export TORCHINDUCTOR_CACHE_DIR="${WORK_ROOT}/torchinductor-cache"
export TRITON_CACHE_DIR="${WORK_ROOT}/triton-cache"
export XDG_CACHE_HOME="${WORK_ROOT}/xdg-cache"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export VLLM_OMNI_USE_QUACK_FP8=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

COMMON_ARGS=(
  --model-root "${MODEL_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --duration "${DURATION_SECONDS}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --seed-base "${SEED_BASE}"
  --num-gpus "${NUM_GPUS}"
  --tensor-parallel-size "${TP_SIZE}"
  --ulysses-degree "${ULYSSES_DEGREE}"
  --ring-degree "${RING_DEGREE}"
  --text-encoder-tp-size "${TEXT_ENCODER_TP_SIZE}"
  --vae-patch-parallel-size "${VAE_PATCH_PARALLEL_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
)
if [[ "${RUN_REF2VA}" == "1" ]]; then
  COMMON_ARGS+=(--expect-ref2va)
fi
if [[ "${ENABLE_DLO}" == "1" ]]; then
  COMMON_ARGS+=(
    --enable-distributed-layerwise-offload
    --dlo-resident-layers "${DLO_RESIDENT_LAYERS}"
  )
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  COMMON_ARGS+=(--enforce-eager)
fi
if [[ -n "${PROFILE_DIR}" ]]; then
  COMMON_ARGS+=(--profiler-dir "${PROFILE_DIR}")
fi
if [[ -n "${FP8_Q_SCALE}" ]]; then COMMON_ARGS+=(--fp8-q-scale "${FP8_Q_SCALE}"); fi
if [[ -n "${FP8_K_SCALE}" ]]; then COMMON_ARGS+=(--fp8-k-scale "${FP8_K_SCALE}"); fi
if [[ -n "${FP8_V_SCALE}" ]]; then COMMON_ARGS+=(--fp8-v-scale "${FP8_V_SCALE}"); fi

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv \
  -lms 500 > "${OUTPUT_DIR}/nvidia-smi.csv" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

RUN_LOG="${OUTPUT_DIR}/run.log"
echo "[1/2] Loading FL2VA and running T2V + first-frame I2V"
"${PYTHON}" "${RUNNER}" --partition fl2va "${COMMON_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"

if [[ "${RUN_REF2VA}" == "1" ]]; then
  echo "[2/2] Loading Ref2VA and running its two reference-conditioned tasks"
  "${PYTHON}" "${RUNNER}" --partition ref2va "${COMMON_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"
else
  echo "[2/2] Ref2VA skipped (set RUN_REF2VA=1 to enable it)"
fi

cleanup
trap - EXIT INT TERM

MEDIA_FILES=(
  "${OUTPUT_DIR}/01_t2va.mp4"
  "${OUTPUT_DIR}/02_fl2va_first_frame.mp4"
)
if [[ "${RUN_REF2VA}" == "1" ]]; then
  MEDIA_FILES+=(
    "${OUTPUT_DIR}/03_ref2va_image_audio.mp4"
    "${OUTPUT_DIR}/04_ref2va_two_videos.mp4"
  )
fi
for media_path in "${MEDIA_FILES[@]}"; do
  if [[ ! -s "${media_path}" ]]; then
    echo "Expected output is missing or empty: ${media_path}" >&2
    exit 1
  fi
  ffprobe -v error \
    -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels,duration \
    -show_entries format=duration,size \
    -of json \
    "${media_path}" > "${media_path%.mp4}.ffprobe.json"
done
sha256sum "${MEDIA_FILES[@]}" > "${OUTPUT_DIR}/artifact_sha256.txt"

awk -F',' -v selected="${CUDA_VISIBLE_DEVICES}" '
  BEGIN {
    count=split(selected, selected_gpu, ",")
    for (i=1; i<=count; i++) wanted[selected_gpu[i]]=1
  }
  NR > 1 {
    gpu=$2
    memory=$3
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu)
    gsub(/[^0-9.]/, "", memory)
    if (wanted[gpu] && memory + 0 > peak[gpu] + 0) peak[gpu]=memory + 0
  }
  END {
    print "physical_gpu,peak_memory_mib"
    for (i=1; i<=count; i++) {
      gpu=selected_gpu[i]
      print gpu "," peak[gpu]
    }
  }
' "${OUTPUT_DIR}/nvidia-smi.csv" > "${OUTPUT_DIR}/gpu_peak_memory.csv"

echo "Completed MiniMax-H3 benchmark tasks."
echo "Outputs: ${OUTPUT_DIR}"
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo "GPU peaks: ${OUTPUT_DIR}/gpu_peak_memory.csv"
