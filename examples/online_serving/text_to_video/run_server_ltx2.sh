#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# LTX-2 online serving startup script with optimization presets.
#
# Usage:
#   bash run_server_ltx2.sh                  # baseline (1 GPU, eager)
#   bash run_server_ltx2.sh ulysses4         # 4-GPU Ulysses SP
#   bash run_server_ltx2.sh cache-dit        # 1 GPU + Cache-DiT
#   bash run_server_ltx2.sh best-combo       # 4-GPU Ulysses SP + Cache-DiT
#
# Online serving benchmarks on H800 (480×768, 41 frames, 20 steps):
#   baseline    : 10.3s inference (1.00×)
#   compile     : ~10.3s warm     (~1.00×) first request +6s warmup
#   ulysses4    : ~10.3s          (~1.00×) no gain at 41 frames
#   cache-dit   :  7.4s avg       (~1.4×)  lossy, variable per request
#   best-combo  :  4.7s avg       (~2.2×)  4-GPU ulysses + cache-dit

set -euo pipefail

MODEL="${MODEL:-Lightricks/LTX-2}"
PORT="${PORT:-8098}"
FLOW_SHIFT="${FLOW_SHIFT:-1.0}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-1.0}"

PRESET="${1:-baseline}"

EXTRA_ARGS=()
case "$PRESET" in
    baseline)
        echo "=== LTX-2 Preset: baseline (1 GPU, enforce-eager) ==="
        EXTRA_ARGS+=(--enforce-eager)
        ;;
    ulysses2)
        echo "=== LTX-2 Preset: 2-GPU Ulysses SP (lossless) ==="
        EXTRA_ARGS+=(--enforce-eager --usp 2)
        ;;
    ulysses4)
        echo "=== LTX-2 Preset: 4-GPU Ulysses SP (lossless) ==="
        EXTRA_ARGS+=(--enforce-eager --usp 4)
        ;;
    cache-dit)
        echo "=== LTX-2 Preset: Cache-DiT (1 GPU, lossy) ==="
        EXTRA_ARGS+=(--enforce-eager --cache-backend cache_dit)
        ;;
    best-combo)
        echo "=== LTX-2 Preset: 4-GPU Ulysses SP + Cache-DiT (best combo) ==="
        EXTRA_ARGS+=(--enforce-eager --usp 4 --cache-backend cache_dit)
        ;;
    compile)
        echo "=== LTX-2 Preset: torch.compile (1 GPU, lossless) ==="
        # torch.compile is the default (no --enforce-eager)
        ;;
    *)
        echo "Usage: $0 {baseline|ulysses2|ulysses4|cache-dit|best-combo|compile}"
        echo ""
        echo "Presets:"
        echo "  baseline    - 1 GPU, eager execution (reference)"
        echo "  ulysses2    - 2-GPU Ulysses SP (lossless)"
        echo "  ulysses4    - 4-GPU Ulysses SP (lossless)"
        echo "  cache-dit   - 1 GPU + Cache-DiT (lossy, ~1.4× speedup)"
        echo "  best-combo  - 4-GPU Ulysses SP + Cache-DiT (~2.2× speedup)"
        echo "  compile     - 1 GPU + torch.compile (slower first request)"
        echo ""
        echo "Environment variables:"
        echo "  MODEL           - Model path (default: Lightricks/LTX-2)"
        echo "  PORT            - Server port (default: 8098)"
        echo "  FLOW_SHIFT      - Scheduler flow shift (default: 1.0)"
        echo "  BOUNDARY_RATIO  - Boundary ratio (default: 1.0)"
        exit 1
        ;;
esac

echo "Model: $MODEL"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
echo "Boundary ratio: $BOUNDARY_RATIO"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --flow-shift "$FLOW_SHIFT" \
    --boundary-ratio "$BOUNDARY_RATIO" \
    "${EXTRA_ARGS[@]}"
