#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${INTERNVLA_A1_RESULT_DIR:-$REPO_ROOT/outputs/internvla_a1/collected_results/$TIMESTAMP}"

: "${INTERNVLA_A1_MODEL_DIR:?Please export INTERNVLA_A1_MODEL_DIR=/path/to/InternVLA-A1-3B-ft-pen}"
: "${INTERNVLA_A1_DATASET_DIR:?Please export INTERNVLA_A1_DATASET_DIR=/path/to/Genie1-Place_Markpen}"
: "${INTERNVLA_A1_PROCESSOR_DIR:?Please export INTERNVLA_A1_PROCESSOR_DIR=/path/to/Qwen3-VL-2B-Instruct}"
: "${INTERNVLA_A1_COSMOS_DIR:?Please export INTERNVLA_A1_COSMOS_DIR=/path/to/Cosmos-Tokenizer-CI8x8-SafeTensor}"

EVAL_OUTPUT_DIR="$RESULT_ROOT/eval_outputs"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"

mkdir -p "$RESULT_ROOT"

export INTERNVLA_A1_MODEL_DIR
export INTERNVLA_A1_DATASET_DIR
export INTERNVLA_A1_PROCESSOR_DIR
export INTERNVLA_A1_COSMOS_DIR

write_env_summary() {
  cat >"$RESULT_ROOT/env_summary.txt" <<EOF
timestamp=$TIMESTAMP
repo_root=$REPO_ROOT
result_root=$RESULT_ROOT
python=$(command -v python || true)
python_version=$(python --version 2>&1 || true)
model_dir=$INTERNVLA_A1_MODEL_DIR
dataset_dir=$INTERNVLA_A1_DATASET_DIR
processor_dir=$INTERNVLA_A1_PROCESSOR_DIR
cosmos_dir=$INTERNVLA_A1_COSMOS_DIR
pwd=$(pwd)
EOF
}

capture_gpu_snapshot() {
  local output_file="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv >"$output_file" 2>&1 || true
  else
    echo "nvidia-smi not found" >"$output_file"
  fi
}

start_gpu_monitor() {
  local output_file="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found" >"$output_file"
    return 1
  fi

  (
    echo "timestamp,index,name,memory.total [MiB],memory.used [MiB],utilization.gpu [%]"
    while true; do
      local now
      now="$(date '+%Y-%m-%d %H:%M:%S')"
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' -v ts="$now" '{gsub(/^[ \t]+|[ \t]+$/, "", $0); print ts "," $0}'
      sleep "$GPU_MONITOR_INTERVAL"
    done
  ) >"$output_file" 2>/dev/null &
  echo $!
}

stop_gpu_monitor() {
  local monitor_pid="${1:-}"
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}

copy_eval_outputs() {
  local source_dir="$1"
  local target_dir="$2"
  local source_real=""
  local target_real=""
  source_real="$(realpath "$source_dir" 2>/dev/null || echo "$source_dir")"
  target_real="$(realpath "$target_dir" 2>/dev/null || echo "$target_dir")"
  if [[ "$source_real" == "$target_real" ]]; then
    return 0
  fi
  mkdir -p "$target_dir"
  if [[ -d "$source_dir" ]]; then
    cp -r "$source_dir"/. "$target_dir"/
  fi
}

run_with_artifacts() {
  local name="$1"
  shift

  local log_file="$RESULT_ROOT/${name}.log"
  local time_file="$RESULT_ROOT/${name}_time.txt"
  local gpu_file="$RESULT_ROOT/${name}_gpu.csv"
  local status_file="$RESULT_ROOT/${name}_status.txt"
  local monitor_pid=""

  monitor_pid="$(start_gpu_monitor "$gpu_file" || true)"
  set +e
  local exit_code=0
  local start_ts=0
  local end_ts=0
  start_ts="$(date +%s)"
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v -o "$time_file" "$@" >"$log_file" 2>&1
    exit_code=$?
    if [[ ! -s "$time_file" ]]; then
      end_ts="$(date +%s)"
      {
        echo "timing_mode=usr_bin_time_empty_fallback"
        echo "start_ts=$start_ts"
        echo "end_ts=$end_ts"
        echo "elapsed_seconds=$((end_ts - start_ts))"
      } >"$time_file"
    fi
  else
    "$@" >"$log_file" 2>&1
    exit_code=$?
    end_ts="$(date +%s)"
    {
      echo "timing_mode=shell_date"
      echo "start_ts=$start_ts"
      echo "end_ts=$end_ts"
      echo "elapsed_seconds=$((end_ts - start_ts))"
    } >"$time_file"
  fi
  set -e
  stop_gpu_monitor "$monitor_pid"

  echo "exit_code=$exit_code" >"$status_file"
  if [[ $exit_code -ne 0 ]]; then
    echo "[error] command failed for $name, see $log_file"
    return $exit_code
  fi
}

write_manifest() {
  cat >"$RESULT_ROOT/README.txt" <<EOF
InternVLA-A1 collected results

Key files:
- env_summary.txt: environment and path summary
- sample_run.log / sample_run_time.txt: one-sample functional run
- forward_benchmark.log / forward_benchmark_time.txt: pure pipeline.forward latency benchmark
- eval_run.log / eval_run_time.txt: GT evaluation run
- pytest_e2e.log / pytest_e2e_time.txt: offline e2e pytest result
- gpu_info_before.csv / gpu_info_after.csv: point-in-time GPU snapshots
- *_gpu.csv: sampled GPU usage during each run
- eval_outputs/: copied output directory from the GT evaluation run

Important outputs:
- forward_benchmark/forward_latency.json
- eval_outputs/summary.json
- eval_outputs/registry/log.json
- eval_outputs/registry/plots/
EOF
}

write_skip_artifact() {
  local name="$1"
  local reason="$2"
  echo "$reason" >"$RESULT_ROOT/${name}.log"
  echo "timing_mode=skipped" >"$RESULT_ROOT/${name}_time.txt"
  echo "exit_code=0" >"$RESULT_ROOT/${name}_status.txt"
  echo "skipped_reason=$reason" >>"$RESULT_ROOT/${name}_status.txt"
  if [[ ! -f "$RESULT_ROOT/${name}_gpu.csv" ]]; then
    echo "skipped,$reason" >"$RESULT_ROOT/${name}_gpu.csv"
  fi
}

write_env_summary
write_manifest
capture_gpu_snapshot "$RESULT_ROOT/gpu_info_before.csv"

run_with_artifacts \
  "sample_run" \
  bash "$ROOT_DIR/run.sh" \
  --output-dir "$RESULT_ROOT/sample_outputs" \
  --num-samples 1 \
  --num-episodes 0

run_with_artifacts \
  "forward_benchmark" \
  python "$ROOT_DIR/end2end.py" \
  --model-dir "$INTERNVLA_A1_MODEL_DIR" \
  --dataset-dir "$INTERNVLA_A1_DATASET_DIR" \
  --benchmark-forward \
  --dtype bfloat16 \
  --attn-implementation eager \
  --warmup-iters 3 \
  --benchmark-iters 10 \
  --output-dir "$RESULT_ROOT/forward_benchmark"

run_with_artifacts \
  "eval_run" \
  bash "$ROOT_DIR/run.sh" \
  --output-dir "$EVAL_OUTPUT_DIR" \
  --num-episodes 1

copy_eval_outputs "$EVAL_OUTPUT_DIR" "$RESULT_ROOT/eval_outputs"

if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)
PY
then
  run_with_artifacts \
    "pytest_e2e" \
    python -m pytest -sv tests/e2e/offline_inference/test_internvla_a1.py -m advanced_model
else
  write_skip_artifact "pytest_e2e" "pytest is not installed in the current python environment"
fi

capture_gpu_snapshot "$RESULT_ROOT/gpu_info_after.csv"

echo "Results written to: $RESULT_ROOT"
