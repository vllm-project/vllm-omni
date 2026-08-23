#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# RTX PRO 5000 SM120: keep each Ulysses-4 group inside one CPU NUMA domain.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,1,5,2,6,3,7}"
export NUM_GPUS=8 TP_SIZE=2 ULYSSES_DEGREE=4 RING_DEGREE=1
export TEXT_ENCODER_TP_SIZE=8 VAE_PATCH_PARALLEL_SIZE=8
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUDNN_ATTN}"
export ENABLE_DLO=0
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}" WARMUP_STEPS="${WARMUP_STEPS:-2}"
export RUN_REF2VA="${RUN_REF2VA:-0}"
exec numactl --interleave=0,1 bash "${SCRIPT_DIR}/run_all_tasks.sh" "$@"
