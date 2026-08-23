#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# RTX PRO 5000 SM120: 1 GPU, DLO is required to fit in 73 GiB.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${DLO_RESIDENT_LAYERS:-20}" == "auto" ]]; then
  exec bash "${SCRIPT_DIR}/run_sm120_dlo_auto.sh" "$@"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS=1 TP_SIZE=1 ULYSSES_DEGREE=1 RING_DEGREE=1
export TEXT_ENCODER_TP_SIZE=1 VAE_PATCH_PARALLEL_SIZE=1
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUDNN_ATTN}"
export ENABLE_DLO=1 DLO_RESIDENT_LAYERS="${DLO_RESIDENT_LAYERS:-20}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}" WARMUP_STEPS="${WARMUP_STEPS:-2}"
export RUN_REF2VA="${RUN_REF2VA:-0}"
NUMA_NODE="${NUMA_NODE:-0}"
exec numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" bash "${SCRIPT_DIR}/run_all_tasks.sh" "$@"
