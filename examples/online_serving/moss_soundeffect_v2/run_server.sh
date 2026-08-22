#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-OpenMOSS-Team/MOSS-SoundEffect-v2.0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"

vllm serve "$MODEL" --omni \
    --host "$HOST" \
    --port "$PORT"
