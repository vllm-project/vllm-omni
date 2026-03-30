#!/usr/bin/env bash
# ── Row 6: Tensor Parallel ───────────────────────────────────────────────────
# Compatibility matrix row 6: tp as baseline, test combinations with cache, sequence parallel, and CFG parallel features.
# Requires ≥2 GPUs (tp gpu_multiplier=2); multi-feature combinations may require 4 GPUs.
#
# Addon combinations to test:
#   teacache      — TP + TeaCache
#   cache_dit     — TP + Cache-DiT
#   ulysses       — TP + Ulysses-SP      (requires 4 GPUs)
#   ring          — TP + Ring-Attn       (requires 4 GPUs)
#   cfg_parallel  — TP + CFG-Parallel    (requires 4 GPUs)
#
# ❌ Known conflict (auto-skipped): tp + hsdp
#
# Note: the framework auto-marks SKIP (GPU) when GPUs are insufficient.
#
# Usage:
#   bash compatibility/scripts/06_tp.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen-Image}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
STEPS="${STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-./compat_results}"
CHARTS="${CHARTS:-1}"

echo "======================================================================"
echo "Row 6 | baseline: tp | addons: teacache cache_dit ulysses ring cfg_parallel"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature tp \
    --addons teacache cache_dit ulysses ring cfg_parallel \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: tp ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/tp" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
