#!/bin/bash
# Helios online serving startup script.
# All three variants (Helios-Base / Helios-Mid / Helios-Distilled) share the same
# server launch; variant-specific knobs are sent per-request via `extra_params`
# (see run_curl_helios.sh).

MODEL="${MODEL:-BestWishYsh/Helios-Base}"
PORT="${PORT:-8098}"

echo "Starting Helios server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
