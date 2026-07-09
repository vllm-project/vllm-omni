#!/usr/bin/env bash
set -euo pipefail

python end2end.py \
  --model aurateam/AURA \
  --deploy-config vllm_omni/deploy/aura_omni.yaml \
  --modalities text,audio \
  "$@"
