#!/bin/bash
# HyperCLOVAX-SEED-Omni-8B Benchmark Script
# Run from vllm-omni root directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
CONCURRENCY="${CONCURRENCY:-1}"
MODE="${MODE:-all}"
OUTPUT_DIR="$SCRIPT_DIR/results"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="$OUTPUT_DIR/benchmark_${MODE}_${TIMESTAMP}.json"

echo "=================================================="
echo "  HyperCLOVAX-SEED-Omni-8B Benchmark"
echo "  BASE_URL    : $BASE_URL"
echo "  MODE        : $MODE"
echo "  NUM_PROMPTS : $NUM_PROMPTS"
echo "  CONCURRENCY : $CONCURRENCY"
echo "  OUTPUT      : $OUTPUT_JSON"
echo "=================================================="

python benchmarks/hcx-omni/benchmark_hcx_omni.py \
    --base-url "$BASE_URL" \
    --mode "$MODE" \
    --num-prompts "$NUM_PROMPTS" \
    --concurrency "$CONCURRENCY" \
    --output-json "$OUTPUT_JSON"

echo ""
echo "Done. Results: $OUTPUT_JSON"
