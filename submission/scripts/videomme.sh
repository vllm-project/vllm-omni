#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VIDEOMME_ROOT=/workspace/Video-MME
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --stage-init-timeout 600 --host 0.0.0.0 --port 8091 --allowed-local-media-path "${VIDEOMME_ROOT}" &
sleep 60
vllm bench serve --omni --port 8091 --max-concurrency 4 --dataset-name videomme --dataset-path /workspace/Video-MME --num-prompts 2700 --trust-remote-code --no-oversample --disable-shuffle --temperature 0 --output-len 128 --videomme-pack-mode minicpm-frames --videomme-max-frames 96 --videomme-duration all --model openbmb/MiniCPM-o-4_5 --endpoint /v1/chat/completions --backend openai-chat-omni --percentile-metrics ttft,tpot,itl,e2el --extra-body "{\"modalities\":[\"text\"],\"chat_template_kwargs\":{\"enable_thinking\":false}}"
