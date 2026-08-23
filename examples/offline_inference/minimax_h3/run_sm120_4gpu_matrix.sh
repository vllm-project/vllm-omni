#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Full 50-step comparison: TP2 x Ulysses2 versus TP4 x Ulysses1.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SCREEN_GPU_COUNT=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,1,3}"
export NUMA_NODE="${NUMA_NODE:-0}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export WARMUP_STEPS="${WARMUP_STEPS:-2}"
export RUN_REF2VA="${RUN_REF2VA:-0}"
exec bash "${SCRIPT_DIR}/run_sm120_topology_screen.sh" "$@"
