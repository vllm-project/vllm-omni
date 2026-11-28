#!/usr/bin/env bash
# Start ONLY this profile's worker and policy frontend, directly as Python modules.
set -euo pipefail
if [[ "${1:-}" == --device ]]; then
  export COSMOS3_DEVICE="${2:?--device requires cuda or npu}"
  shift 2
fi
for arg in "$@"; do
  case "$arg" in
    --port|--port=*|--host|--host=*|--vllm-base-url|--vllm-base-url=*)
      echo "Use COSMOS3_POLICY_PORT/COSMOS3_HOST; this launcher manages its own worker" >&2
      exit 2 ;;
  esac
done
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VLLM_OMNI_ROOT"
if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
  exec "$VLLM_PYTHON" -m vllm_omni.entrypoints.cosmos3.policy_server --help
fi
export COSMOS3_MODEL="${COSMOS3_MODEL:-$VLLM_OMNI_ROOT/models/cosmos3-edge-libero10-diffusers}"
export COSMOS3_WORKER_PORT="${COSMOS3_WORKER_PORT:-8100}"
export COSMOS3_POLICY_PORT="${COSMOS3_POLICY_PORT:-8101}"
export COSMOS3_LOG_DIR="${COSMOS3_LOG_DIR:-$VLLM_OMNI_ROOT/outputs/cosmos3_baseline}"
[[ -x "$VLLM_PYTHON" ]] || { echo "Python missing: $VLLM_PYTHON" >&2; exit 2; }
[[ -f "$COSMOS3_MODEL/model_index.json" ]] || { echo "Diffusers model missing: $COSMOS3_MODEL" >&2; exit 2; }
mkdir -p "$COSMOS3_LOG_DIR"
exec 9>"$COSMOS3_LOG_DIR/server.lock"
flock -n 9 || { echo "A server already owns $COSMOS3_LOG_DIR" >&2; exit 2; }
for port in "$COSMOS3_WORKER_PORT" "$COSMOS3_POLICY_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Port $port is in use; choose different COSMOS3_WORKER_PORT/COSMOS3_POLICY_PORT" >&2
    exit 2
  fi
done
export VLLM_OMNI_SERVER_STORAGE__PATH="$COSMOS3_LOG_DIR/storage"
mkdir -p "$VLLM_OMNI_SERVER_STORAGE__PATH"
export COSMOS3_EXECUTION_TIMING="${COSMOS3_EXECUTION_TIMING:-1}"
export COSMOS3_TIMING_SYNC="${COSMOS3_TIMING_SYNC:-0}"
export COSMOS3_WORKER_TIMELINE="$COSMOS3_LOG_DIR/worker_timeline.jsonl"
export COSMOS3_SERVER_TIMELINE="$COSMOS3_LOG_DIR/server_timeline.jsonl"
worker_pid=''
policy_pid=''
cleanup() {
  trap - EXIT INT TERM
  for pid in "$policy_pid" "$worker_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -f "$COSMOS3_LOG_DIR/server.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "$$" > "$COSMOS3_LOG_DIR/server.pid"
echo "baseline server: $COSMOS3_DEVICE; source: $VLLM_OMNI_ROOT; Python: $VLLM_PYTHON"
echo "Logs: $COSMOS3_LOG_DIR/worker.log and $COSMOS3_LOG_DIR/policy.log"
setsid "$VLLM_PYTHON" -u -m vllm_omni.entrypoints.cli.main \
  serve "$COSMOS3_MODEL" --omni --host 127.0.0.1 --port "$COSMOS3_WORKER_PORT" \
  --init-timeout 1800 --no-guardrails "${COSMOS3_WORKER_ARGS[@]}" \
  >"$COSMOS3_LOG_DIR/worker.log" 2>&1 &
worker_pid=$!
deadline=$((SECONDS + ${COSMOS3_STARTUP_TIMEOUT:-1800}))
until curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$COSMOS3_WORKER_PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$worker_pid" 2>/dev/null || ((SECONDS >= deadline)); then
    tail -n 60 "$COSMOS3_LOG_DIR/worker.log" >&2
    exit 1
  fi
  sleep 1
done
setsid "$VLLM_PYTHON" -u -m vllm_omni.entrypoints.cosmos3.policy_server \
  --vllm-base-url "http://127.0.0.1:$COSMOS3_WORKER_PORT" \
  --host "${COSMOS3_HOST:-127.0.0.1}" --port "$COSMOS3_POLICY_PORT" \
  --model "$COSMOS3_MODEL" --seed 0 --flow-shift 1.0 --poll-interval 0.01 \
  --action-time-schedule constant_euler --disable-streaming "$@" \
  >"$COSMOS3_LOG_DIR/policy.log" 2>&1 &
policy_pid=$!
echo "Policy URL: http://${COSMOS3_HOST:-127.0.0.1}:$COSMOS3_POLICY_PORT (frontend starting)"
# Exit and stop this launcher's children if either server fails.
wait -n "$worker_pid" "$policy_pid"
exit 1
