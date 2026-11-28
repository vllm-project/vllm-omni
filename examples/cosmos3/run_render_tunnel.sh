#!/usr/bin/env bash
# Render transport only. Run on 168; no model or evaluation is launched here.
set -uo pipefail
local_port="${COSMOS3_RENDER_PORT:-19960}"
remote_port="${COSMOS3_RENDER_FORWARD_PORT:-$local_port}"
tunnel_pid=''
cleanup() {
  trap - INT TERM
  [[ -z "$tunnel_pid" ]] || kill -TERM "$tunnel_pid" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM
while true; do
  ssh -NT -S none -o ControlMaster=no -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
    -R "127.0.0.1:$remote_port:127.0.0.1:$local_port" "$@" &
  tunnel_pid=$!
  wait "$tunnel_pid"
  status=$?
  echo "$(date -u +%FT%TZ) Render SSH exited ($status); reconnecting in 2s" >&2
  tunnel_pid=''
  sleep 2
done
