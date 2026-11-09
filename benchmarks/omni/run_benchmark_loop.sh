#!/bin/bash
set -euo pipefail

# =====================================================================
# DTPS Benchmark Loop Script
# Runs omni_benchmark_serving.py across multiple seeds and collects
# results into a labelled output directory for side-by-side comparison.
#
# Usage:
#   ./run_benchmark_loop.sh [OPTIONS]
#
# Examples:
#   # 10 rounds, default params
#   ./run_benchmark_loop.sh -r 10 -l dtps
#
#   # Custom model, host, port, task mix
#   ./run_benchmark_loop.sh -r 10 -l baseline \
#       --model /models/hunyuan-image3 \
#       --host 127.0.0.1 --port 8098 \
#       --num-i2t 14 --num-t2i 3 --num-it2i 3
#
#   # Specific seeds only
#   ./run_benchmark_loop.sh -l debug -s 4 9 10
#
# Any extra args after the parsed options are forwarded verbatim to
# omni_benchmark_serving.py (e.g. --stream, --disable-tqdm, etc.).
# =====================================================================

# ---- Defaults (overridable via flags) ----
ROUNDS=0
LABEL="baseline"
SEEDS=""
HOST="127.0.0.1"
PORT=8098
MODEL="/models/hunyuan-image3"
NUM_I2T=14
NUM_T2I=3
NUM_IT2I=3
MAX_CONCURRENCY=20
INTERVAL=2
OUTPUT_ROOT="benchmark-res"
EXTRA_ARGS=""

# ---- Parse options ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--rounds)    ROUNDS="$2"; shift 2 ;;
        -l|--label)     LABEL="$2"; shift 2 ;;
        -s|--seeds)     shift; while [[ $# -gt 0 && "$1" != -* ]]; do SEEDS="$SEEDS $1"; shift; done ;;
        --host)         HOST="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --num-i2t)      NUM_I2T="$2"; shift 2 ;;
        --num-t2i)      NUM_T2I="$2"; shift 2 ;;
        --num-it2i)     NUM_IT2I="$2"; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
        --interval)     INTERVAL="$2"; shift 2 ;;
        --output-root)  OUTPUT_ROOT="$2"; shift 2 ;;
        --)             shift; EXTRA_ARGS="$@"; break ;;
        -*)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---- Resolve seed list ----
if [[ -z "$SEEDS" ]]; then
    if [[ "$ROUNDS" -lt 1 ]]; then
        echo "Error: provide --rounds N or --seeds <seed...>"
        exit 1
    fi
    SEEDS=$(seq 1 "$ROUNDS")
fi

SEED_LIST=($SEEDS)
TOTAL=${#SEED_LIST[@]}

echo "Starting benchmark loop: label=$LABEL  seeds=[$(echo $SEEDS)]  total=$TOTAL"
echo "  model=$MODEL  host=$HOST:$PORT"
echo "  mix: i2t=$NUM_I2T  t2i=$NUM_T2I  it2i=$NUM_IT2I  concurrency=$MAX_CONCURRENCY"
[[ -n "$EXTRA_ARGS" ]] && echo "  extra: $EXTRA_ARGS"
echo ""

# ---- Run ----
idx=0
for seed in "${SEED_LIST[@]}"; do
    idx=$((idx + 1))
    echo "=========================================="
    echo "[$idx/$TOTAL] label=$LABEL  seed=$seed"
    echo "=========================================="

    python3 benchmarks/omni/omni_benchmark_serving.py \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --num-i2t "$NUM_I2T" \
        --num-t2i "$NUM_T2I" \
        --num-it2i "$NUM_IT2I" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --shuffle \
        --it2i-endpoint images-edits \
        --gen-resolution-weights "512x512=1,1024x1024=1,1280x720=1" \
        --randomize-input \
        --return-stage-metrics \
        --output-dir "${OUTPUT_ROOT}/${LABEL}/omni-${NUM_I2T}-${NUM_T2I}-${NUM_IT2I}-seed-${seed}" \
        --seed "$seed" \
        $EXTRA_ARGS

    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "WARNING: seed=$seed exited with code $rc"
    fi

    if [[ $idx -lt $TOTAL ]]; then
        echo "seed=$seed done. Waiting ${INTERVAL}s..."
        sleep "$INTERVAL"
    fi
done

echo ""
echo "All $TOTAL rounds completed (label=$LABEL)."
echo "Results: ${OUTPUT_ROOT}/${LABEL}/"
