#!/usr/bin/env bash
# ── Row 5: CFG-Parallel ──────────────────────────────────────────────────────
# Compatibility matrix row 5: cfg_parallel as baseline, test combinations with cache and sequence parallel features.
# Requires ≥2 GPUs (cfg_parallel gpu_multiplier=2).
#
# Addon combinations to test:
#   teacache   — CFG-Parallel + TeaCache
#   cache_dit  — CFG-Parallel + Cache-DiT
#   ulysses    — CFG-Parallel + Ulysses-SP      (requires 4 GPUs)
#   ring       — CFG-Parallel + Ring-Attn       (requires 4 GPUs)
#
# Note: ulysses / ring combinations require 4 GPUs; the framework auto-marks SKIP (GPU) if insufficient.
#
# Usage:
#   bash compatibility/scripts/05_cfg_parallel.sh
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
echo "Row 5 | baseline: cfg_parallel | addons: teacache cache_dit ulysses ring"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature cfg_parallel \
    --addons teacache cache_dit ulysses ring \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: cfg_parallel ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/cfg_parallel" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
