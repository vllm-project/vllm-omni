#!/usr/bin/env bash
# ── Row 3: Ulysses-SP ────────────────────────────────────────────────────────
# Compatibility matrix row 3: ulysses as baseline, test combinations with cache acceleration features.
# Requires ≥2 GPUs (ulysses gpu_multiplier=2).
#
# ❌ Known conflict (auto-skipped): ulysses + teacache (SP and TeaCache are not compatible, probably due to bugs in the implementation)
#
# Addon combinations to test:
#   cache_dit  — Ulysses + Cache-DiT
#
# Usage:
#   bash compatibility/scripts/03_ulysses.sh
#   MODEL=Tongyi-MAI/Z-Image-Turbo NUM_PROMPTS=10 bash compatibility/scripts/03_ulysses.sh
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
echo "Row 3 | baseline: ulysses | addons: cache_dit"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature ulysses \
    --addons cache_dit \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: ulysses ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/ulysses" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
