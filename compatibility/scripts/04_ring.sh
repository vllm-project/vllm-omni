#!/usr/bin/env bash
# ── Row 4: Ring-Attn ─────────────────────────────────────────────────────────
# Compatibility matrix row 4: ring as baseline, test combinations with cache acceleration features.
# Requires ≥2 GPUs (ring gpu_multiplier=2).
#
# ✅ Already verified: ring + ulysses (marked ✅ in the matrix; not re-run here)
#
# ❌ Known conflict (auto-skipped): ring + teacache (SP and TeaCache are not compatible, probably due to bugs in the implementation)
#
# Addon combinations to test:
#   cache_dit  — Ring + Cache-DiT
#
# Usage:
#   bash compatibility/scripts/04_ring.sh
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
echo "Row 4 | baseline: ring | addons: cache_dit"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature ring \
    --addons cache_dit \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: ring ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/ring" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
