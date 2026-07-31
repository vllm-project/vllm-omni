#!/usr/bin/env bash
# Restart the duplex server and wait until it is serving.
#
# No longer required between conversations -- multiple sessions per boot work.
# Still the right first move when debugging: a wedged pipeline accepts audio
# and produces nothing, so anything measured after a wedge is worthless.
set -euo pipefail
NAME="${1:-qwen-duplex-test}"
PORT="${2:-8099}"
docker restart "$NAME" >/dev/null
printf 'restarting %s' "$NAME"
until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/health")" = "200" ]; do
  docker ps --format '{{.Names}}' | grep -q "^${NAME}$" || { echo " -- container exited"; exit 1; }
  printf '.'; sleep 10
done
echo " -- ready on :${PORT}"
