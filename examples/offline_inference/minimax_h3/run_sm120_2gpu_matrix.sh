#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Full 50-step comparison: two-GPU topology and DLO resident-layer candidates.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SCREEN_GPU_COUNT=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUMA_NODE="${NUMA_NODE:-0}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export WARMUP_STEPS="${WARMUP_STEPS:-2}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export RUN_REF2VA="${RUN_REF2VA:-0}"
exec bash "${SCRIPT_DIR}/run_sm120_topology_screen.sh" "$@"
