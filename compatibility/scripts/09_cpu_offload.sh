#!/usr/bin/env bash
# ── CPU Offload (Module-level) ────────────────────────────────────────────────
# Uses cpu_offload (module-level CPU offloading: DiT + text encoder offloaded as whole modules) as baseline,
# testing compatibility with cache acceleration and parallelism features.
#
# Note: cpu_offload is not listed in SINGLE_CARD_ONLY, so the framework will not auto-exclude multi-GPU combos;
#       whether they actually run depends on hardware. The framework auto-marks SKIP (GPU) if GPUs are insufficient.
#
# Addon combinations to test:
#   teacache      — CPU Offload + TeaCache
#   cache_dit     — CPU Offload + Cache-DiT
#   ulysses       — CPU Offload + Ulysses-SP      (requires 2 GPUs)
#   ring          — CPU Offload + Ring-Attn       (requires 2 GPUs)
#   cfg_parallel  — CPU Offload + CFG-Parallel    (requires 2 GPUs)
#   tp            — CPU Offload + Tensor Parallel (requires 2 GPUs)
#   hsdp          — CPU Offload + HSDP            (requires 2 GPUs)
#
# ❌ Known conflict (auto-skipped): cpu_offload + layerwise_offload
#
# Usage:
#   bash compatibility/scripts/09_cpu_offload.sh
#   MODEL=Tongyi-MAI/Z-Image-Turbo bash compatibility/scripts/09_cpu_offload.sh
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
echo "CPU Offload (module-level) | baseline: cpu_offload"
echo "addons: teacache cache_dit ulysses ring cfg_parallel tp hsdp"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature cpu_offload \
    --addons teacache cache_dit ulysses ring cfg_parallel tp hsdp \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: cpu_offload ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/cpu_offload" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
