#!/bin/bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
DEVICE_IDS="${DEVICE_IDS:-0,1,2,3,4,5,6,7}"
BACKEND_START_PORT="${BACKEND_START_PORT:-8091}"
BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-/tmp/wan22_t2v_super_p95_usp8}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-910B3}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1000000}"
BACKEND_HEALTH_TIMEOUT_S="${BACKEND_HEALTH_TIMEOUT_S:-1800}"
BACKEND_HEALTH_POLL_INTERVAL_S="${BACKEND_HEALTH_POLL_INTERVAL_S:-10}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
QUOTA_EVERY="${QUOTA_EVERY:-20}"
QUOTA_AMOUNT="${QUOTA_AMOUNT:-1}"
THRESHOLD_RATIO="${THRESHOLD_RATIO:-0.8}"
SACRIFICIAL_LOAD_FACTOR="${SACRIFICIAL_LOAD_FACTOR:-0.1}"
CLEAN_BACKEND_LOG_DIR="${CLEAN_BACKEND_LOG_DIR:-1}"
VLLM_OMNI_MASTER_PORT="${VLLM_OMNI_MASTER_PORT:-30191}"

if [[ "${CLEAN_BACKEND_LOG_DIR}" == "1" ]]; then
  rm -rf "${BACKEND_LOG_DIR}"
fi
mkdir -p "${BACKEND_LOG_DIR}"

exec env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/super_p95_dispatcher.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-servers 1 \
  --device-ids "${DEVICE_IDS}" \
  --model "${MODEL}" \
  --backend-start-port "${BACKEND_START_PORT}" \
  --backend-hardware-profiles "${HARDWARE_PROFILE}" \
  --backend-scheduler super_p95_step \
  --quota-every "${QUOTA_EVERY}" \
  --quota-amount "${QUOTA_AMOUNT}" \
  --threshold-ratio "${THRESHOLD_RATIO}" \
  --sacrificial-load-factor "${SACRIFICIAL_LOAD_FACTOR}" \
  --backend-log-dir "${BACKEND_LOG_DIR}" \
  --backend-env "VLLM_OMNI_MASTER_PORT=${VLLM_OMNI_MASTER_PORT}" \
  --request-timeout-s "${REQUEST_TIMEOUT_S}" \
  --backend-health-timeout-s "${BACKEND_HEALTH_TIMEOUT_S}" \
  --backend-health-poll-interval-s "${BACKEND_HEALTH_POLL_INTERVAL_S}" \
  --backend-args=--omni \
  --backend-args=--usp \
  --backend-args=8 \
  --backend-args=--enable-layerwise-offload \
  --backend-args=--boundary-ratio \
  --backend-args="${BOUNDARY_RATIO}" \
  --backend-args=--flow-shift \
  --backend-args="${FLOW_SHIFT}" \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling
