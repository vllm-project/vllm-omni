#!/bin/bash
# 启动 vLLM-Omni 服务（官方固定启动命令兼容版）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /models/MiniCPM-o-4_5 --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --stage-init-timeout 600 \
  --host 0.0.0.0 --port 8091
