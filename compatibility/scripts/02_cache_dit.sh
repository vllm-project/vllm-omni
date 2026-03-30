#!/usr/bin/env bash
# ── Row 2: Cache-DiT ─────────────────────────────────────────────────────────
# Compatibility matrix row 2: smoke test with cache_dit as the baseline only.
# No addon combinations to test in this row (teacache conflicts with cache_dit; they are mutually exclusive).
#
# Usage:
#   bash compatibility/scripts/02_cache_dit.sh
#   MODEL=Tongyi-MAI/Z-Image-Turbo bash compatibility/scripts/02_cache_dit.sh
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
echo "Row 2 | baseline: cache_dit | addons: (none)"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature cache_dit \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: cache_dit ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/cache_dit" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
