#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-quay.io/ascend/vllm-omni:v0.18.0}"
CONTAINER="${CONTAINER:-ax-vllm-qwen-8inst-ingress-strategy}"
MODEL_DIR="${MODEL_DIR:-/docker/models/Qwen-Image}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-18291}"
PROXY_PORT="${PROXY_PORT:-28093}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/docker/aixuan/vllm-omni-v0.18.0-test}"
CTR_ROOT="/vllm-workspace/vllm-omni"
PROXY_LOG="${PROXY_LOG:-/tmp/rr_proxy_timing_${PROXY_PORT}.jsonl}"

# Strategy defaults for branch handoff:
# if OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES is not set,
# plugin auto-computes budget by OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE.
OMNI_INGRESS_BATCH_DRR_ENABLE="${OMNI_INGRESS_BATCH_DRR_ENABLE:-1}"
OMNI_INGRESS_BATCH_CAPS="${OMNI_INGRESS_BATCH_CAPS:-{\"512x512_20\":4,\"768x768_20\":4,\"1024x1024_25\":1,\"1536x1536_35\":1}}"
OMNI_INGRESS_DRR_MAX_WAIT_MS="${OMNI_INGRESS_DRR_MAX_WAIT_MS:-800}"
OMNI_INGRESS_DRR_STRICT_BATCHING="${OMNI_INGRESS_DRR_STRICT_BATCHING:-0}"
OMNI_INGRESS_DRR_Q_BASE="${OMNI_INGRESS_DRR_Q_BASE:-12}"
OMNI_INGRESS_DRR_AGE_THRESHOLD_MS="${OMNI_INGRESS_DRR_AGE_THRESHOLD_MS:-2000}"
OMNI_INGRESS_DRR_AGE_BONUS_FACTOR="${OMNI_INGRESS_DRR_AGE_BONUS_FACTOR:-1.0}"
OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE="${OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE:-weight_inv_cost}"
OMNI_INGRESS_DRR_REQUEST_WEIGHTS="${OMNI_INGRESS_DRR_REQUEST_WEIGHTS:-}"
OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES="${OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES:-}"

log() { echo "[$(date '+%F %T')] $*"; }

if [[ ! -d "${MODEL_DIR}" ]]; then
  log "ERROR: model dir missing: ${MODEL_DIR}"
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_INSTANCES="${#GPUS[@]}"
if [[ "${NUM_INSTANCES}" -ne 8 ]]; then
  log "WARN: GPU_LIST gives ${NUM_INSTANCES} instances, expected 8."
fi

docker rm -f "${CONTAINER}" 2>/dev/null || true

port_map=()
dev_map=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  port="$((BASE_PORT + i))"
  port_map+=(-p "${port}:${port}")
  dev_map+=(--device "/dev/davinci${gpu}")
done
port_map+=(-p "${PROXY_PORT}:${PROXY_PORT}")

log "Starting strategy container ${CONTAINER}"
docker run -d --name "${CONTAINER}" \
  --shm-size=4g --ipc=host \
  "${dev_map[@]}" \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "${MODEL_DIR}:/model:ro" \
  "${port_map[@]}" \
  "${IMAGE}" sleep infinity >/dev/null

log "Copy plugin and patch scripts into container"
docker exec "${CONTAINER}" bash -lc "mkdir -p ${CTR_ROOT}/server_ingress_plugin"
docker cp "${WORKSPACE_DIR}/image_ingress_scheduler_single.py" \
  "${CONTAINER}:${CTR_ROOT}/server_ingress_plugin/image_ingress_scheduler_single.py"
docker cp "${WORKSPACE_DIR}/patch_api_server_ingress_singlefile.py" \
  "${CONTAINER}:/tmp/patch_api_server_ingress_singlefile.py"
if [ -f "${WORKSPACE_DIR}/rr_vllm_http_proxy_timing.py" ]; then
  docker cp "${WORKSPACE_DIR}/rr_vllm_http_proxy_timing.py" "${CONTAINER}:/tmp/rr_vllm_http_proxy_timing.py"
else
  log "Proxy script not found at ${WORKSPACE_DIR}/rr_vllm_http_proxy_timing.py; installing no-op fallback in container"
  docker exec "${CONTAINER}" bash -lc "cat > /tmp/rr_vllm_http_proxy_timing.py <<'PY'
#!/usr/bin/env python3
import sys

print('rr_vllm_http_proxy_timing.py not found in workspace; skipping proxy startup', file=sys.stderr)
PY
chmod +x /tmp/rr_vllm_http_proxy_timing.py"
fi
log "Patch api_server.py inside container (runtime only, image unchanged)"
docker exec "${CONTAINER}" bash -lc "cd ${CTR_ROOT} && python3 /tmp/patch_api_server_ingress_singlefile.py"

log "Sequentially starting ${NUM_INSTANCES} strategy instances"
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  port="$((BASE_PORT + i))"
  log "Starting strategy instance $((i + 1))/${NUM_INSTANCES}: gpu=${gpu}, port=${port}"

  docker exec "${CONTAINER}" bash -lc "\
    export ASCEND_RT_VISIBLE_DEVICES=${gpu}; \
    export VLLM_WORKER_MULTIPROC_METHOD=spawn; \
    export PYTHONPATH=${CTR_ROOT}:\${PYTHONPATH:-}; \
    export OMNI_INGRESS_BATCH_DRR_ENABLE='${OMNI_INGRESS_BATCH_DRR_ENABLE}'; \
    export OMNI_INGRESS_BATCH_CAPS='${OMNI_INGRESS_BATCH_CAPS}'; \
    export OMNI_INGRESS_DRR_MAX_WAIT_MS='${OMNI_INGRESS_DRR_MAX_WAIT_MS}'; \
    export OMNI_INGRESS_DRR_STRICT_BATCHING='${OMNI_INGRESS_DRR_STRICT_BATCHING}'; \
    export OMNI_INGRESS_DRR_Q_BASE='${OMNI_INGRESS_DRR_Q_BASE}'; \
    export OMNI_INGRESS_DRR_AGE_THRESHOLD_MS='${OMNI_INGRESS_DRR_AGE_THRESHOLD_MS}'; \
    export OMNI_INGRESS_DRR_AGE_BONUS_FACTOR='${OMNI_INGRESS_DRR_AGE_BONUS_FACTOR}'; \
    export OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE='${OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE}'; \
    export OMNI_INGRESS_DRR_REQUEST_WEIGHTS='${OMNI_INGRESS_DRR_REQUEST_WEIGHTS}'; \
    export OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES='${OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES}'; \
    cd ${CTR_ROOT}; \
    nohup python3 -m vllm_omni.entrypoints.cli.main serve /model --omni --port ${port} \
      --tensor-parallel-size 1 --vae-use-tiling --vae-use-slicing \
      > /tmp/vllm_${port}.log 2>&1 &"

  docker exec "${CONTAINER}" bash -lc "\
    ok=0; \
    for j in \$(seq 1 360); do \
      if curl -sS http://127.0.0.1:${port}/health >/dev/null 2>&1; then ok=1; break; fi; \
      sleep 2; \
    done; \
    if [ \"\$ok\" != \"1\" ]; then echo not_ready_${port}; tail -120 /tmp/vllm_${port}.log; exit 1; fi"
done

backends=""
for i in "${!GPUS[@]}"; do
  port="$((BASE_PORT + i))"
  backends+="http://127.0.0.1:${port},"
done
backends="${backends%,}"

log "Starting timing RR proxy on ${PROXY_PORT}"
docker exec "${CONTAINER}" bash -lc "\
  : > '${PROXY_LOG}'; \
  nohup python3 /tmp/rr_vllm_http_proxy_timing.py --backends '${backends}' --port ${PROXY_PORT} --log-file '${PROXY_LOG}' \
    > /tmp/rr_proxy_${PROXY_PORT}.log 2>&1 &"

docker exec "${CONTAINER}" bash -lc "\
  ok=0; \
  for i in \$(seq 1 120); do \
    if curl -sS http://127.0.0.1:${PROXY_PORT}/health >/dev/null 2>&1; then ok=1; break; fi; \
    sleep 1; \
  done; \
  if [ \"\$ok\" != \"1\" ]; then echo proxy_not_ready; tail -120 /tmp/rr_proxy_${PROXY_PORT}.log; exit 1; fi"

log "READY: strategy env is up"
log "Container: ${CONTAINER}"
log "Proxy URL: http://127.0.0.1:${PROXY_PORT}/v1/images/generations"
