#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --stage-init-timeout 600 --host 0.0.0.0 --port 8091 &
sleep 60
python /workspace/vllm-omni/examples/online_serving/minicpmo/gradio_demo.py --minicpmo45-api-base http://localhost:8091/v1 --minicpmo45-model openbmb/MiniCPM-o-4_5 --port 7862
