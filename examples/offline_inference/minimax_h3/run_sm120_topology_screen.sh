#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Screen only the SM120 topology choices that cannot be inferred from B300.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../../" && pwd)}"
SCREEN_GPU_COUNT="${SCREEN_GPU_COUNT:-4}"
SCREEN_ROOT="${SCREEN_ROOT:-${WORK_ROOT}/results/sm120-topology-screen-$(date -u +%Y%m%dT%H%M%SZ)}"

export ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUDNN_ATTN}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-2}"
export RUN_REF2VA="${RUN_REF2VA:-0}"
FAILED_CASES=()

run_case() {
  local label="$1" cuda_visible_devices="$2" numa_mode="$3" tp="$4" ulysses="$5" enable_dlo="$6" resident_layers="$7"
  local output_dir="${SCREEN_ROOT}/${label}"
  echo "=== ${label}: TP${tp} x Ulysses${ulysses} ==="
  if ! env \
    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    NUM_GPUS="$((tp * ulysses))" \
    TP_SIZE="${tp}" ULYSSES_DEGREE="${ulysses}" RING_DEGREE=1 \
    TEXT_ENCODER_TP_SIZE="$((tp * ulysses))" \
    VAE_PATCH_PARALLEL_SIZE="$((tp * ulysses))" \
    ENABLE_DLO="${enable_dlo}" DLO_RESIDENT_LAYERS="${resident_layers}" \
    OUTPUT_DIR="${output_dir}" \
    bash -c "${numa_mode} bash \"${SCRIPT_DIR}/run_all_tasks.sh\""; then
    FAILED_CASES+=("${label}")
    echo "=== ${label} failed; continuing with remaining candidates ===" >&2
  fi
}

mkdir -p "${SCREEN_ROOT}"
case "${SCREEN_GPU_COUNT}" in
  1)
    run_case "tp1-u1-dlo-r20" "${CUDA_VISIBLE_DEVICES:-0}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 1 1 1 20
    run_case "tp1-u1-dlo-r35" "${CUDA_VISIBLE_DEVICES:-0}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 1 1 1 35
    run_case "tp1-u1-dlo-r50" "${CUDA_VISIBLE_DEVICES:-0}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 1 1 1 50
    ;;
  2)
    run_case "tp1-u2-dlo-r20" "${CUDA_VISIBLE_DEVICES:-0,1}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 1 2 1 20
    run_case "tp1-u2-dlo-r35" "${CUDA_VISIBLE_DEVICES:-0,1}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 1 2 1 35
    run_case "tp1-u2-dlo-r50" "${CUDA_VISIBLE_DEVICES:-0,1}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 1 2 1 50
    run_case "tp2-u1-dlo-r20" "${CUDA_VISIBLE_DEVICES:-0,1}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 2 1 1 20
    run_case "tp2-u1-dlo-r35" "${CUDA_VISIBLE_DEVICES:-0,1}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 2 1 1 35
    run_case "tp2-u1-dlo-r50" "${CUDA_VISIBLE_DEVICES:-0,1}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 2 1 1 50
    ;;
  4)
    run_case "tp2-u2-resident" "${CUDA_VISIBLE_DEVICES:-0,2,1,3}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 2 2 0 0
    run_case "tp4-u1-resident" "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "numactl --cpunodebind=${NUMA_NODE:-0} --membind=${NUMA_NODE:-0}" 4 1 0 0
    ;;
  8)
    run_case "tp2-u4-resident" "${CUDA_VISIBLE_DEVICES:-0,4,1,5,2,6,3,7}" "numactl --interleave=0,1" 2 4 0 0
    run_case "tp4-u2-resident" "${CUDA_VISIBLE_DEVICES:-0,4,1,5,2,6,3,7}" "numactl --interleave=0,1" 4 2 0 0
    run_case "tp8-u1-resident" "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" "numactl --interleave=0,1" 8 1 0 0
    ;;
  *)
    echo "SCREEN_GPU_COUNT must be one of: 1, 2, 4, 8" >&2
    exit 2
    ;;
esac

if (( ${#FAILED_CASES[@]} )); then
  printf '%s\n' "${FAILED_CASES[@]}" > "${SCREEN_ROOT}/failed_cases.txt"
  echo "Completed screen with failed cases: ${FAILED_CASES[*]}"
else
  echo "Completed screen: ${SCREEN_ROOT}"
fi
