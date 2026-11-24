#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run on 168 with nohup; own one renderer and its reconnecting SSH tunnel.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
gpu="${1:?Usage: run_render_background.sh GPU PORT OUTPUT_DIR [SSH_ARGS...] }"
port="${2:?port required}"
output="${3:?output directory required}"
shift 3
[[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ ]] || exit 2
((port > 1024 && port < 65536)) || exit 2
if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
  echo "Renderer port is already occupied: $port" >&2
  exit 2
fi
output="$(realpath -m "$output")"
mkdir -p "$output"
exec 8>"$output/supervisor.lock"
flock -n 8 || { echo "Renderer supervisor already running" >&2; exit 2; }
[[ ! -e "$output/gpu_renderer_proof.json" && ! -e "$output/render_server_stages.jsonl" ]] || { echo "Use a fresh renderer output directory" >&2; exit 2; }
renderer_pid=''
tunnel_pid=''
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "$tunnel_pid" "$renderer_pid"; do
    if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
  printf '%s\n' "$status" >"$output/exit_code"
  rm -f "$output/supervisor.pid"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "$$" >"$output/supervisor.pid"
export MUJOCO_EGL_DEVICE_ID="$gpu" COSMOS3_RENDER_PORT="$port" COSMOS3_RENDER_OUT="$output"
export COSMOS3_RENDER_FORWARD_PORT="$port"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash "$root/examples/cosmos3/run_renderer.sh" >"$output/renderer.log" 2>&1 &
renderer_pid=$!
deadline=$((SECONDS+90))
until curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >"$output/health.json"; do
  kill -0 "$renderer_pid" || { tail -30 "$output/renderer.log"; exit 1; }
  ((SECONDS < deadline)) || { echo "Renderer startup timed out" >&2; exit 1; }
  sleep 1
done
bash "$root/examples/cosmos3/run_render_tunnel.sh" "$@" >"$output/tunnel.log" 2>&1 &
tunnel_pid=$!
wait -n "$renderer_pid" "$tunnel_pid"
exit 1
