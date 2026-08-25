#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --stage-init-timeout 600 --host 0.0.0.0 --port 8091 &
sleep 60
vllm bench serve --omni --port 8091 --trust-remote-code --max-concurrency 1 --num-warmup 3 --dataset-name seed-tts --dataset-path /workspace/seedtts_testset --num-prompts 32 --no-oversample --seed-tts-wer-eval --seed-tts-wer-save-items --model openbmb/MiniCPM-o-4_5 --endpoint /v1/chat/completions --backend openai-chat-omni --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf --extra-body "{\"modalities\":[\"text\",\"audio\"],\"chat_template_kwargs\":{\"enable_thinking\":false,\"use_tts_template\":true}}"
