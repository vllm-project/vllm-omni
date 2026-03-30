#!/usr/bin/env bash
# ── Row 9: VAE Patch Parallel ─────────────────────────────────────────────────
# Compatibility matrix row 9: vae_patch_parallel is an addon-only feature and cannot be used as a standalone baseline.
#
# This script stacks vae_patch_parallel on top of each parallel baseline in separate rounds;
# after each round it immediately calls analyze_compat_results.py on the results.
# Each round also includes teacache / cache_dit as additional addons to cover three-way combinations.
#
# Addon combinations to test (vae_patch_parallel as addon):
#   cfg_parallel  + vae_patch_parallel [+ teacache / cache_dit]
#   ulysses       + vae_patch_parallel [+ teacache / cache_dit]
#   ring          + vae_patch_parallel [+ teacache / cache_dit]
#   tp            + vae_patch_parallel [+ teacache / cache_dit]
#   hsdp          + vae_patch_parallel [+ teacache / cache_dit]
#
# ❌ Known conflict (auto-skipped): vae_patch_parallel + layerwise_offload
#
# Note: all parallel baselines require ≥2 GPUs; multi-parallel combinations may require 4 GPUs.
#
# Usage:
#   bash compatibility/scripts/10_vae_patch_parallel.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen-Image}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
STEPS="${STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-./compat_results}"
CHARTS="${CHARTS:-1}"

PARALLEL_BASELINES=(cfg_parallel ulysses ring tp hsdp)

for BASELINE in "${PARALLEL_BASELINES[@]}"; do
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/vae_patch_parallel_on_${BASELINE}"

    echo "======================================================================"
    echo "Row 9 | baseline: ${BASELINE} | addons: vae_patch_parallel teacache cache_dit"
    echo "======================================================================"

    python "${SCRIPT_DIR}/../run_compat_test.py" \
        --baseline-feature "${BASELINE}" \
        --addons vae_patch_parallel teacache cache_dit \
        --model "${MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --steps "${STEPS}" \
        --output-dir "${RUN_OUTPUT_DIR}"

    echo ""
    echo "--- Analyzing results: ${BASELINE} (vae_patch_parallel run) ---"
    python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
        --results-dir "${RUN_OUTPUT_DIR}/${BASELINE}" \
        ${CHARTS:+--charts} \
        || echo "[WARN] analyze_compat_results returned non-zero (check results above)"

    echo ""
done

echo "======================================================================"
echo "Row 9 complete — all vae_patch_parallel combination runs finished."
echo "======================================================================"
