#!/bin/bash
# Gradio Demo 启动脚本
nohup python /vllm-workspace/vllm-omni/examples/online_serving/minicpmo/gradio_demo.py \
  --minicpmo45-api-base http://localhost:8091/v1 \
  --minicpmo45-model openbmb/MiniCPM-o-4_5 \
  --port 7862 \
  --host 0.0.0.0 \
  > /workspace/logs/gradio_demo.log 2>&1 &
