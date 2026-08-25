#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DAILY_OMNI_VIDEOS=/workspace/Daily-Omni/Videos
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --stage-init-timeout 600 --host 0.0.0.0 --port 8091 --allowed-local-media-path "${DAILY_OMNI_VIDEOS}" --interleave-mm-strings &
sleep 60
vllm bench serve --omni --port 8091 --max-concurrency 10 --dataset-name daily-omni --num-prompts 1197 --trust-remote-code --no-oversample --temperature 0 --output-len 128 --daily-omni-input-mode all --daily-omni-pack-mode minicpm-interleave --daily-omni-video-dir /workspace/Daily-Omni/Videos --daily-omni-qa-json /workspace/Daily-Omni/qa.json --model openbmb/MiniCPM-o-4_5 --endpoint /v1/chat/completions --backend openai-chat-omni --percentile-metrics ttft,tpot,itl,e2el --extra-body "{\"modalities\":[\"text\"],\"chat_template_kwargs\":{\"enable_thinking\":false}}"
