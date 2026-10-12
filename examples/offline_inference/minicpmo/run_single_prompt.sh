#!/usr/bin/env bash
# Single-prompt offline smoke test (text → text + 24 kHz speech).
set -euo pipefail
cd "$(dirname "$0")"

python end2end.py --output-wav output_audio \
                  --query-type text
