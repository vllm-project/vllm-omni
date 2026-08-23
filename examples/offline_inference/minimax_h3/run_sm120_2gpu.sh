#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# RTX PRO 5000 SM120: one PXB pair; DLO is required for 72 GiB cards.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS=2 TP_SIZE=2 ULYSSES_DEGREE=1 RING_DEGREE=1
export TEXT_ENCODER_TP_SIZE=2 VAE_PATCH_PARALLEL_SIZE=2
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUDNN_ATTN}"
export ENABLE_DLO=1 DLO_RESIDENT_LAYERS="${DLO_RESIDENT_LAYERS:-35}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}" WARMUP_STEPS="${WARMUP_STEPS:-2}"
export RUN_REF2VA="${RUN_REF2VA:-0}"
NUMA_NODE="${NUMA_NODE:-0}"
exec numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" bash "${SCRIPT_DIR}/run_all_tasks.sh" "$@"
