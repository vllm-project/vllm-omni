#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Serve independent diffusion replicas, with Wan2.2 TI2V-5B as the default.
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
NUM_REPLICAS="${NUM_REPLICAS-1}"
DEVICES="${DEVICES-0}"   # one GPU per replica for tensor_parallel_size=1

if [[ ! "$NUM_REPLICAS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_REPLICAS must be a positive integer" >&2
    exit 1
fi
if [[ ! "$DEVICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    echo "DEVICES must be comma-separated non-negative GPU ordinals" >&2
    exit 1
fi
IFS=',' read -r -a DEVICE_IDS <<< "$DEVICES"
if [[ "${#DEVICE_IDS[@]}" != "$NUM_REPLICAS" ]]; then
    echo "DEVICES must contain exactly one GPU ordinal per replica" >&2
    exit 1
fi
declare -A SEEN_DEVICES=()
for device in "${DEVICE_IDS[@]}"; do
    if [[ -n "${SEEN_DEVICES[$device]+present}" ]]; then
        echo "DEVICES must contain unique GPU ordinals" >&2
        exit 1
    fi
    SEEN_DEVICES[$device]=1
done

HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${DEPLOY_TEMPLATE:-$HERE/wan2_2_ti2v_dp.yaml}"
if [[ ! -f "$TEMPLATE" ]]; then
    echo "Deploy template not found: $TEMPLATE" >&2
    exit 1
fi
CFG="$(mktemp --suffix=_replica_dp.yaml)"
trap 'rm -f "$CFG"' EXIT

sed -e "s/__NUM_REPLICAS__/${NUM_REPLICAS}/" \
    -e "s/__DEVICES__/${DEVICES}/" \
    "$TEMPLATE" > "$CFG"

echo "Serving ${MODEL}: num_replicas=${NUM_REPLICAS} devices=${DEVICES} (config ${CFG})"

vllm serve "$MODEL" --omni \
    --host "$HOST" \
    --port "$PORT" \
    --deploy-config "$CFG" "$@"
