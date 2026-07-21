#!/usr/bin/env bash
# Multi-prompt offline batch (py_generator mode).
set -euo pipefail
cd "$(dirname "$0")"

python end2end.py --output-wav output_audio \
                  --query-type text \
                  --txt-prompts ../qwen3_omni/text_prompts_10.txt \
                  --py-generator
