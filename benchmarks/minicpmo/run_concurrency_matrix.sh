#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/root/autodl-tmp/models/MiniCPM-o-4_5}
PYTHON=${PYTHON:-/root/autodl-tmp/envs/vllm-omni/bin/python}
VLLM=${VLLM:-/root/autodl-tmp/envs/vllm-omni/bin/vllm}
PORT=${PORT:-8099}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/logs/minicpmo-concurrency/matrix}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-0}
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-4}
REQUESTS_PER_GROUP=${REQUESTS_PER_GROUP:-}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH="$(dirname "$PYTHON"):$PATH"

SERVER_PID=
SAMPLER_PID=

stop_processes() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$SAMPLER_PID" ]]; then
    kill -TERM -- "-$SAMPLER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
  SAMPLER_PID=
}
trap stop_processes EXIT INT TERM

run_config() {
  local label=$1
  local config=$2
  local run_id
  local run_dir
  run_id="${label}_$(date -u +%Y%m%dT%H%M%SZ)"
  run_dir="$OUTPUT_ROOT/$run_id"
  mkdir -p "$run_dir"
  export VLLM_OMNI_CONCURRENCY_TRACE_PATH="$run_dir/trace.jsonl"
  export VLLM_OMNI_CONCURRENCY_TRACE_RUN_ID="$run_id"

  "$PYTHON" -m vllm_omni.metrics.concurrency_trace snapshot-config --stage-config "$config"
  setsid "$PYTHON" -m vllm_omni.metrics.concurrency_trace sample \
    --devices 0,1 \
    --metrics-url "http://127.0.0.1:$PORT/metrics" \
    --interval-s 0.5 >"$run_dir/sampler.log" 2>&1 &
  SAMPLER_PID=$!
  echo "$SAMPLER_PID" >"$run_dir/sampler.pid"

  setsid "$VLLM" serve "$MODEL" --omni \
    --deploy-config "$config" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" >"$run_dir/server.log" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$run_dir/server.pid"

  local ready=false
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
      ready=true
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 5
  done
  if [[ "$ready" != true ]]; then
    echo "Server failed to become healthy for $label" | tee "$run_dir/status.txt"
    tail -100 "$run_dir/server.log"
    return 1
  fi

  if [[ "$WARMUP_REQUESTS" -gt 0 ]]; then
    "$PYTHON" "$REPO_ROOT/benchmarks/minicpmo/benchmark_concurrency.py" \
      --model "$MODEL" \
      --concurrency "$WARMUP_CONCURRENCY" \
      --requests "$WARMUP_REQUESTS" \
      --max-tokens 48 \
      --trace-path "$run_dir/trace.jsonl" \
      --output-dir "$run_dir/warmup"
  fi

  local -a concurrency_levels
  read -r -a concurrency_levels <<<"${CONCURRENCIES:-1 2 4}"
  for concurrency in "${concurrency_levels[@]}"; do
    local requests
    if [[ -n "$REQUESTS_PER_GROUP" ]]; then
      requests=$REQUESTS_PER_GROUP
    else
      requests=$concurrency
      if [[ "$concurrency" -gt 1 ]]; then
        requests=$((concurrency * 2))
      fi
    fi
    "$PYTHON" "$REPO_ROOT/benchmarks/minicpmo/benchmark_concurrency.py" \
      --model "$MODEL" \
      --concurrency "$concurrency" \
      --requests "$requests" \
      --max-tokens 48 \
      --trace-path "$run_dir/trace.jsonl" \
      --output-dir "$run_dir/c$concurrency"
  done
  echo PASS >"$run_dir/status.txt"
  stop_processes
}

mkdir -p "$OUTPUT_ROOT"
CONFIG_LABELS=("$@")
if [[ ${#CONFIG_LABELS[@]} -eq 0 ]]; then
  CONFIG_LABELS=(
    max_seqs_1_1
    max_seqs_1_2
    max_seqs_2_1
    max_seqs_2_2
    max_seqs_1_4
    max_seqs_4_1
    max_seqs_4_4
  )
fi
for label in "${CONFIG_LABELS[@]}"; do
  config="$REPO_ROOT/benchmarks/minicpmo/configs/$label.yaml"
  if [[ ! -f "$config" ]]; then
    echo "Unknown config label: $label" >&2
    exit 2
  fi
  run_config "$label" "$config"
done
