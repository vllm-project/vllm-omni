#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Wan-AI/Wan2.2-T2V-A14B-Diffusers"}
PORT=${PORT:-8093}

vllm serve "$MODEL" --omni --port "$PORT"
