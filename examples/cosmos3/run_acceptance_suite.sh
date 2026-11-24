#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# One paired suite on one NPU. Each profile owns its server and output directory.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suite="${1:?Usage: run_acceptance_suite.sh SUITE NPU PORT cpu_OR_RENDER_URL OUTPUT_ROOT}"
export COSMOS3_NPU_DEVICES="${2:?NPU required}"
port="${3:?worker port required}"
render_url="${4:?cpu or render URL required}"
output="${5:?output root required}"
case "$suite" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo "Unsupported suite: $suite" >&2; exit 2;; esac
[[ "$port" =~ ^[0-9]+$ && "$COSMOS3_NPU_DEVICES" =~ ^[0-9]+$ ]] || exit 2
case "$render_url" in cpu|http://*|https://*) ;; *) echo "Expected cpu or renderer URL" >&2; exit 2;; esac
[[ "${COSMOS3_TRIALS:-3}" =~ ^[1-9][0-9]*$ ]] || { echo "COSMOS3_TRIALS must be positive" >&2; exit 2; }
source "$root/examples/cosmos3/env_native.sh"
cd "$root"
output="$(realpath -m "$output")"
mkdir -p "$output/$suite"
exec 8>"$output/$suite/pair.lock"
flock -n 8 || { echo "A paired evaluation owns this suite" >&2; exit 2; }
server_pid=''
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for profile in faster base; do
  if [[ "$profile" == faster ]]; then
    model="${COSMOS3_FASTER_MODEL:-$root/models/f0/diffusers}"
    server=run_server_faster.sh
    client=run_client_stream.sh
  else
    model="${COSMOS3_BASE_MODEL:-$root/models/b0/diffusers}"
    server=run_server.sh
    client=run_client.sh
  fi
  [[ ! -e "$output/$suite/$profile" ]] || { echo "Refusing to overwrite $output/$suite/$profile" >&2; exit 2; }
  export COSMOS3_MODEL="$model" COSMOS3_WORKER_PORT="$port" COSMOS3_POLICY_PORT="$((port+1))"
  export COSMOS3_LOG_DIR="$output/$suite/${profile}_server"
  export COSMOS3_SERVER_URL="http://127.0.0.1:$COSMOS3_POLICY_PORT"
  bash "examples/cosmos3/$server" --device npu >"$output/$suite/${profile}_launcher.log" 2>&1 &
  server_pid=$!
  deadline=$((SECONDS + 1800))
  until curl --noproxy '*' -fsS --max-time 2 "$COSMOS3_SERVER_URL/info" >"$output/$suite/${profile}_info.json" 2>/dev/null; do
    kill -0 "$server_pid" 2>/dev/null || { tail -40 "$output/$suite/${profile}_launcher.log"; exit 1; }
    (( SECONDS < deadline )) || { echo "Server readiness timed out" >&2; exit 1; }
    sleep 1
  done
  if [[ "$render_url" == cpu ]]; then
    export COSMOS3_RENDER=cpu
    unset REMOTE_RENDER_URL
  else
    export COSMOS3_RENDER=remote REMOTE_RENDER_URL="$render_url"
  fi
  export EVAL_OUT="$output/$suite/$profile"
  bash "examples/cosmos3/$client" --task_suite "$suite" \
    --task_ids 0,1,2,3,4,5,6,7,8,9 --num_trials_per_task "${COSMOS3_TRIALS:-3}" \
    --exclude_dummy_time --timing_warmup_requests 0 >"$output/$suite/${profile}_client.log" 2>&1
  cleanup
  server_pid=''
done
printf 'complete\n' > "$output/$suite/COMPLETE"
