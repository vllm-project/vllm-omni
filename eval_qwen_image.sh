#!/usr/bin/env bash
set -euo pipefail

# Qwen-Image server launch examples for this script.
#
# Notes:
# - This script only launches the client benchmark. Start the server manually first.
# - Use the v0.18.0-super-p95 dispatcher for both baseline and super_p95 so the
#   serving topology stays identical.
# Baseline, 8x 910B3:
# env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
# python3 benchmarks/diffusion/super_p95_dispatcher.py \
#   --host 127.0.0.1 \
#   --port 8080 \
#   --num-servers 8 \
#   --device-ids 0,1,2,3,4,5,6,7 \
#   --model Qwen/Qwen-Image \
#   --backend-start-port 8091 \
#   --backend-hardware-profiles 910B3 \
#   --backend-scheduler step_baseline \
#   --backend-args=--omni \
#   --backend-args=--vae-use-slicing \
#   --backend-args=--vae-use-tiling
#
# Super-P95, 8x 910B3:
# env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
# python3 benchmarks/diffusion/super_p95_dispatcher.py \
#   --host 127.0.0.1 \
#   --port 8080 \
#   --num-servers 8 \
#   --device-ids 0,1,2,3,4,5,6,7 \
#   --model Qwen/Qwen-Image \
#   --backend-start-port 8091 \
#   --backend-hardware-profiles 910B3 \
#   --backend-scheduler super_p95_step \
#   --quota-every 20 \
#   --quota-amount 1 \
#   --threshold-ratio 0.8 \
#   --sacrificial-load-factor 0.1 \
#   --backend-args=--omni \
#   --backend-args=--vae-use-slicing \
#   --backend-args=--vae-use-tiling

usage() {
  cat <<'EOF'
Usage:
  bash eval_qwen_image.sh baseline
  bash eval_qwen_image.sh super_p95

Description:
  Run the Qwen-Image client benchmark against an already running server.
  This script does not launch the server. Start the matching baseline or super_p95
  server manually first, then run this script.

Environment overrides:
  BASE_URL          Default: http://127.0.0.1:8080
  MODEL             Default: Qwen/Qwen-Image
  NUM_PROMPTS       Default: 500
  MAX_CONCURRENCY   Default: 1000
  REQUEST_RATES     Default: "0.8"
  WARMUP_REQUESTS   Default: 1
  WARMUP_STEPS      Default: 1
  SEED              Default: 0
  RANDOM_REQUEST_SEED Default: 8
  ARRIVAL_SEED      Default: 8
  CLIENT_TIMEOUT_S  Default: 1000000
  OUTPUT_ROOT       Default: benchmarks/diffusion/results/qwen_image_p95_sweep
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

RUN_LABEL="$1"
case "${RUN_LABEL}" in
  baseline|super_p95)
    ;;
  *)
    echo "Invalid mode: ${RUN_LABEL}" >&2
    usage
    exit 1
    ;;
esac

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-Qwen/Qwen-Image}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1000}"
REQUEST_RATES="${REQUEST_RATES:-0.8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
SEED="${SEED:-0}"
RANDOM_REQUEST_SEED="${RANDOM_REQUEST_SEED:-8}"
ARRIVAL_SEED="${ARRIVAL_SEED:-8}"
CLIENT_TIMEOUT_S="${CLIENT_TIMEOUT_S:-1000000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmarks/diffusion/results/qwen_image_p95_sweep}"

DEFAULT_QWEN_IMAGE_RANDOM4="$(cat <<'EOF'
[
  {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
  {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
  {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
  {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
]
EOF
)"
QWEN_IMAGE_RANDOM4="${QWEN_IMAGE_RANDOM4:-$DEFAULT_QWEN_IMAGE_RANDOM4}"

OUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"
mkdir -p "${OUT_DIR}"

echo "Run label       : ${RUN_LABEL}"
echo "Base URL        : ${BASE_URL}"
echo "Model           : ${MODEL}"
echo "Num prompts     : ${NUM_PROMPTS}"
echo "Max concurrency : ${MAX_CONCURRENCY}"
echo "Request rates   : ${REQUEST_RATES}"
echo "Arrival seed    : ${ARRIVAL_SEED}"
echo "Output dir      : ${OUT_DIR}"
echo "Canonical cmd   : BASE_URL=${BASE_URL} NUM_PROMPTS=${NUM_PROMPTS} MAX_CONCURRENCY=${MAX_CONCURRENCY} REQUEST_RATES=${REQUEST_RATES} SEED=${SEED} RANDOM_REQUEST_SEED=${RANDOM_REQUEST_SEED} ARRIVAL_SEED=${ARRIVAL_SEED} bash eval_qwen_image.sh ${RUN_LABEL}"

python - <<'PY' <<<"${QWEN_IMAGE_RANDOM4}" >/dev/null
import json
import sys
json.loads(sys.stdin.read())
PY

HEALTH_CODE="$(curl --noproxy '*' -s -o /tmp/eval_qwen_image_health.json -w '%{http_code}' "${BASE_URL}/health" || true)"
if [[ "${HEALTH_CODE}" != "200" ]]; then
  echo "Server health check failed for ${BASE_URL}/health (status=${HEALTH_CODE})." >&2
  cat /tmp/eval_qwen_image_health.json 2>/dev/null || true
  exit 1
fi

for rate in ${REQUEST_RATES}; do
  output_file="${OUT_DIR}/rate_${rate}.json"
  echo
  echo "=== rate=${rate} -> ${output_file} ==="
  env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --backend vllm-omni \
    --dataset random \
    --task t2i \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${rate}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --warmup-num-inference-steps "${WARMUP_STEPS}" \
    --client-timeout-s "${CLIENT_TIMEOUT_S}" \
    --seed "${SEED}" \
    --random-request-seed "${RANDOM_REQUEST_SEED}" \
    --arrival-seed "${ARRIVAL_SEED}" \
    --random-request-config "${QWEN_IMAGE_RANDOM4}" \
    --output-file "${output_file}"
done

echo
echo "Finished. Results written to ${OUT_DIR}"
