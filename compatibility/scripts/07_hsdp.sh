#!/usr/bin/env bash
# ── Row 7: HSDP ──────────────────────────────────────────────────────────────
# Compatibility matrix row 7: hsdp as baseline, test combinations with cache, sequence parallel, and CFG parallel features.
# Requires ≥2 GPUs (hsdp gpu_multiplier=2); multi-feature combinations may require 4 GPUs.
#
# Addon combinations to test:
#   teacache      — HSDP + TeaCache
#   cache_dit     — HSDP + Cache-DiT
#   ulysses       — HSDP + Ulysses-SP      (requires 4 GPUs)
#   ring          — HSDP + Ring-Attn       (requires 4 GPUs)
#   cfg_parallel  — HSDP + CFG-Parallel    (requires 4 GPUs)
#
# ❌ Known conflict (auto-skipped): hsdp + tp
#
# Note: the framework auto-marks SKIP (GPU) when GPUs are insufficient.
#
# Usage:
#   bash compatibility/scripts/07_hsdp.sh
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
echo "Row 7 | baseline: hsdp | addons: teacache cache_dit ulysses ring cfg_parallel"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature hsdp \
    --addons teacache cache_dit ulysses ring cfg_parallel \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: hsdp ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/hsdp" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
