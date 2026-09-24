#!/usr/bin/env bash
# Serve N independent Wan2.2 TI2V-5B replicas (replica data parallelism).
#
# Usage:
#   NUM_REPLICAS=1 DEVICES=0        ./run_server.sh      # baseline
#   NUM_REPLICAS=2 DEVICES=0,1      ./run_server.sh
#   NUM_REPLICAS=4 DEVICES=0,1,2,3  ./run_server.sh
#
# Then drive load with bench_replica_dp.py (see README.md) and compare
# throughput across replica counts.
set -euo pipefail

MODEL="${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8098}"
NUM_REPLICAS="${NUM_REPLICAS:-1}"
DEVICES="${DEVICES:-0}"   # one GPU per replica for tensor_parallel_size=1

HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$HERE/wan2_2_ti2v_dp.yaml"
CFG="$(mktemp --suffix=_wan2_2_dp.yaml)"
trap 'rm -f "$CFG"' EXIT

sed -e "s/__NUM_REPLICAS__/${NUM_REPLICAS}/" \
    -e "s/__DEVICES__/${DEVICES}/" \
    "$TEMPLATE" > "$CFG"

echo "Serving ${MODEL}: num_replicas=${NUM_REPLICAS} devices=${DEVICES} (config ${CFG})"

vllm serve "$MODEL" --omni \
    --host "$HOST" \
    --port "$PORT" \
    --deploy-config "$CFG"
