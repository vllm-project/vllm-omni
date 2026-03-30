#!/usr/bin/env bash
# ── Row 8: CPU Offloading (Layerwise) ────────────────────────────────────────
# Compatibility matrix row 8: layerwise_offload as baseline, test combinations with cache acceleration features.
# Single-GPU only (SINGLE_CARD_ONLY); all multi-GPU features are marked ❌ and auto-skipped by the framework.
#
# Addon combinations to test:
#   teacache   — Layerwise Offload + TeaCache
#   cache_dit  — Layerwise Offload + Cache-DiT
#
# ❌ Known conflicts (auto-skipped): layerwise_offload + ulysses/ring/cfg_parallel/tp/hsdp
#
# Usage:
#   bash compatibility/scripts/08_layerwise_offload.sh
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
echo "Row 8 | baseline: layerwise_offload | addons: teacache cache_dit"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature layerwise_offload \
    --addons teacache cache_dit \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: layerwise_offload ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/layerwise_offload" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
