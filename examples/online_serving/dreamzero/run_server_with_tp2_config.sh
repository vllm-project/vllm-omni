#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES=0,1 vllm serve GEAR-Dreams/DreamZero-DROID --omni \
  --host 127.0.0.1 --port 8000 \
  --served-model-name dreamzero-droid \
  --enforce-eager --disable-log-stats \
  --stage-configs-path vllm_omni/model_executor/stage_configs/dreamzero_tp2_cfg1.yaml