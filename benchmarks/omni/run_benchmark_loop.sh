#!/bin/bash

# Usage: ./run_benchmark_loop.sh <rounds>
# Example: ./run_benchmark_loop.sh 5

ROUNDS=${1:-1}
INTERVAL=1

if [[ "$ROUNDS" -lt 1 ]]; then
    echo "Error: rounds must be >= 1"
    exit 1
fi

echo "Starting $ROUNDS rounds of benchmark testing..."

for i in $(seq 1 "$ROUNDS"); do
    echo "=========================================="
    echo "Round $i / $ROUNDS  (seed=$i)"
    echo "=========================================="

    python3 benchmarks/omni/omni_benchmark_serving.py \
        --host 127.0.0.1 \
        --port 8098 \
        --model /workspace/models/weight/HunyuanImage-3.0-Instruct \
        --num-i2t 14 \
        --num-t2i 3 \
        --num-it2i 3 \
        --max-concurrency 20 \
        --shuffle \
        --it2i-endpoint images-edits \
        --gen-resolution-weights "512x512=1,1024x1024=1,1280x720=1" \
        --randomize-input \
        --output-dir "benchmark-res/baseline/omni-14-3-3-seed-$i" \
        --seed "$i" \
	--dry-run

    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "WARNING: Round $i exited with code $exit_code"
    fi

    if [[ $i -lt $ROUNDS ]]; then
        echo ""
        echo "Round $i completed. Waiting ${INTERVAL}s before next round..."
        sleep "$INTERVAL"
    fi
done

echo ""
echo "All $ROUNDS rounds completed."
