#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# One persistent model server; four sequential suites on one NPU.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
profile="${1:?Usage: run_serial_profile.sh base_OR_faster NPU WORKER_PORT RENDER_URL OUTPUT_ROOT [TRIALS]}"
device="${2:?NPU required}"
port="${3:?worker port required}"
render_url="${4:?GPU renderer URL required}"
output="${5:?output root required}"
trials="${6:-50}"
case "$profile" in base|faster) ;; *) echo "Profile must be base or faster" >&2; exit 2;; esac
[[ "$device" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ && "$trials" =~ ^[1-9][0-9]*$ ]] || exit 2
((port > 1024 && port < 65535)) || { echo "Invalid worker port" >&2; exit 2; }
case "$render_url" in http://*|https://*) ;; *) echo "Expected GPU renderer URL" >&2; exit 2;; esac
source "$root/examples/cosmos3/env_native.sh"
cd "$root"
output="$(realpath -m "$output")"
mkdir -p "$output"
exec 8>"$output/$profile.lock"
flock -n 8 || { echo "Profile already running: $profile" >&2; exit 2; }
exec 7>"/tmp/cosmos3-serial-npu-$device.lock"
flock -n 7 || { echo "Another serial evaluation owns NPU $device" >&2; exit 2; }
suites=(libero_spatial libero_object libero_goal libero_10)
for suite in "${suites[@]}"; do
  [[ ! -e "$output/$suite/$profile" ]] || { echo "Refusing to overwrite $output/$suite/$profile" >&2; exit 2; }
done
[[ ! -e "$output/${profile}_server" ]] || { echo "Refusing to reuse server logs" >&2; exit 2; }
if [[ "$profile" == base ]]; then
  export COSMOS3_MODEL="${COSMOS3_BASE_MODEL:-$root/models/b0/diffusers}"
  server=run_server.sh
  client=run_client.sh
else
  export COSMOS3_MODEL="${COSMOS3_FASTER_MODEL:-$root/models/f0/diffusers}"
  server=run_server_faster.sh
  client=run_client_stream.sh
fi
export COSMOS3_NPU_DEVICES="$device" COSMOS3_WORKER_PORT="$port" COSMOS3_POLICY_PORT="$((port+1))"
export COSMOS3_LOG_DIR="$output/${profile}_server"
export COSMOS3_SERVER_URL="http://127.0.0.1:$COSMOS3_POLICY_PORT"
export COSMOS3_RENDER=remote REMOTE_RENDER_URL="$render_url"
export NUMBA_DISABLE_JIT=0 NUMBA_OPT=3 COSMOS3_DTYPE=float32 COSMOS3_STRICT_PRECISION=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
# The original launchers choose server/client OpenMP defaults of 2/1.
unset OMP_NUM_THREADS
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$root/.runtime/numba-cache-serial-opt3}"
server_pid=''
client_pid=''
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$client_pid" ]]; then kill -TERM "$client_pid" 2>/dev/null || true; wait "$client_pid" 2>/dev/null || true; fi
  if [[ -n "$server_pid" ]]; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
  printf '%s\n' "$status" >"$output/$profile.exit_code"
  rm -f "$output/$profile.pid"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "$$" >"$output/$profile.pid"

# Validate all initial-state indices before loading the model.
"$COSMOS3_CLIENT_PYTHON" - "$trials" "$profile" "$output" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from vllm_omni_cosmos3_client.closed_loop_eval import _configure_mujoco_env, _import_libero
from vllm_omni_cosmos3_client import closed_loop_eval as evaluator

trials, profile, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
_configure_mujoco_env("osmesa")
_import_libero()
counts = {}
for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    suite = evaluator.benchmark.get_benchmark_dict()[name]()
    assert suite.n_tasks == 10
    counts[name] = [len(suite.get_task_init_states(task)) for task in range(10)]
    assert min(counts[name]) >= trials, (name, counts[name], trials)
model = Path(os.environ["COSMOS3_MODEL"])
assert (model / "model_index.json").is_file(), model
report = dict(profile=profile, trials_per_task=trials, episodes_per_profile=40*trials,
              suite_order=list(counts), initial_state_counts=counts,
              model=str(model), npu=os.environ["COSMOS3_NPU_DEVICES"],
              renderer=os.environ["REMOTE_RENDER_URL"],
              environment={k: os.environ.get(k) for k in
                           ("NUMBA_DISABLE_JIT", "NUMBA_OPT", "NUMBA_CACHE_DIR", "COSMOS3_MESA_ROOT")})
root = Path.cwd()
paths = ["examples/cosmos3/run_serial_profile.sh", "examples/cosmos3/env_native.sh",
         "plugins/cosmos3_client/vllm_omni_cosmos3_client/closed_loop_eval.py",
         "vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py"]
report["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
(output / (profile + ".manifest.json")).write_text(json.dumps(report, indent=2) + "\n")
PY
curl --noproxy '*' -fsS --max-time 10 "$render_url/health" >"$output/$profile.renderer.json"
"$COSMOS3_CLIENT_PYTHON" - "$output/$profile.renderer.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
assert info["service"] == "render_only_v1" and info["physics_execution"] is False, info
assert info["mujoco_version"] == "2.3.7", info
PY
mkdir -p "$COSMOS3_LOG_DIR"
bash "examples/cosmos3/$server" --device npu >"$COSMOS3_LOG_DIR/launcher.log" 2>&1 &
server_pid=$!
deadline=$((SECONDS + 1800))
until curl --noproxy '*' -fsS --max-time 2 "$COSMOS3_SERVER_URL/info" >"$output/$profile.info.json" 2>/dev/null; do
  kill -0 "$server_pid" 2>/dev/null || { tail -40 "$COSMOS3_LOG_DIR/launcher.log"; exit 1; }
  ((SECONDS < deadline)) || { echo "Model startup timed out" >&2; exit 1; }
  sleep 2
done
for suite in "${suites[@]}"; do
  kill -0 "$server_pid"
  mkdir -p "$output/$suite"
  export EVAL_OUT="$output/$suite/$profile"
  echo "$(date -u +%FT%TZ) START $profile $suite NPU=$device trials=$trials"
  bash "examples/cosmos3/$client" --task_suite "$suite" \
    --task_ids 0,1,2,3,4,5,6,7,8,9 --num_trials_per_task "$trials" \
    --num_envs 1 --seed 0 --action_horizon 16 --exclude_dummy_time \
    --timing_warmup_requests 0 >"$output/$suite/${profile}_client.log" 2>&1 &
  client_pid=$!
  wait "$client_pid"
  client_pid=''
  "$COSMOS3_CLIENT_PYTHON" - "$EVAL_OUT/summary.json" "$trials" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
trials = int(sys.argv[2])
assert summary["total_episodes"] == 10*trials and summary["num_trials_per_task"] == trials
assert summary["selected_task_ids"] == list(range(10))
assert all(len(task["episode_results"]) == trials for task in summary["task_results"])
assert all(not episode.get("error") for task in summary["task_results"] for episode in task["episode_results"])
throughput = summary["throughput"]
raw = throughput["evaluation_wall"]
adjusted = throughput["render_excluded"]["evaluation_no_render_no_dummy"]
window = throughput["render_excluded"]["complete_windows_no_render"]
print(
    f"THROUGHPUT suite={summary['task_suite']} "
    f"raw_actions_per_s={raw['executed_actions_per_s']:.6f} "
    f"no_render_no_dummy_actions_per_s={adjusted['actions_per_s']:.6f} "
    f"complete_window_no_render_actions_per_s={window['actions_per_s']:.6f} "
    f"executed_actions={adjusted['executed_actions']} "
    f"raw_s={raw['elapsed_s']:.6f} "
    f"render_excluded_s={adjusted['render_excluded_s']:.6f} "
    f"dummy_s={adjusted['dummy_stabilization_s']:.6f} "
    f"total_excluded_union_s={adjusted['total_excluded_s']:.6f} "
    f"effective_s={adjusted['elapsed_no_render_no_dummy_s']:.6f}",
    flush=True,
)
PY
  printf 'complete\n' >"$output/$suite/$profile.COMPLETE"
  echo "$(date -u +%FT%TZ) COMPLETE $profile $suite"
done
printf 'complete\n' >"$output/$profile.COMPLETE"
