#!/usr/bin/env bash
# Single-prompt offline run with thinker tensor-parallel (3-GPU layout).
# Thinker on GPU 0,1 (TP=2); talker + Token2Wav on GPU 2.
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(cd ../../.. && pwd)"

python end2end.py --output-wav output_audio \
                  --query-type use_audio \
                  --deploy-config "${REPO_ROOT}/vllm_omni/deploy/minicpmo_4_5_3gpu.yaml" \
                  --stage-init-timeout 300
