#!/usr/bin/env bash
set -euo pipefail

# Start a local MediaMTX RTSP server (listens on :8554 by default).
# Download the binary + mediamtx.yml from https://github.com/bluenviron/mediamtx/releases
# and either keep them next to this script or point the vars below at them.

MEDIAMTX_BIN="${MEDIAMTX_BIN:-./mediamtx}"
MEDIAMTX_CONFIG="${MEDIAMTX_CONFIG:-./mediamtx.yml}"

exec "$MEDIAMTX_BIN" "$MEDIAMTX_CONFIG"
