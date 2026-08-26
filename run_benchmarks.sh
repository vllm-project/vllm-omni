#!/bin/bash
# 三个 Benchmark 全量执行脚本（与本次实测命令一致）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0

# 1. Seed-TTS 中文全量（WER/SIM 精度）
export SEED_TTS_WAVLM_MODEL=/workspace/wavlm-base-plus
export SEED_TTS_EVAL_DEVICE=cpu
export SEED_TTS_SIM_DEVICE=cpu
SEED_TTS_SIM_EVAL=1 SEED_TTS_WER_EVAL=1 vllm bench serve \
  --omni --port 8091 --trust-remote-code \
  --max-concurrency 1 --num-warmups 2 \
  --dataset-name seed-tts --dataset-path /workspace/seedtts_testset \
  --seed-tts-locale zh --num-prompts 2020 --disable-shuffle --no-oversample \
  --seed-tts-wer-eval --seed-tts-wer-save-items \
  --model openbmb/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' \
  --tokenizer /models/MiniCPM-o-4_5

# 2. Daily-Omni 全量精度
vllm bench serve \
  --omni --port 8091 --max-concurrency 10 \
  --dataset-name daily-omni --num-prompts 1197 --trust-remote-code \
  --no-oversample --temperature 0 --output-len 512 \
  --daily-omni-input-mode all --daily-omni-pack-mode minicpm-interleave \
  --daily-omni-video-dir /workspace/Daily-Omni/Videos \
  --daily-omni-qa-json /workspace/Daily-Omni/qa.json \
  --model openbmb/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el \
  --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' \
  --tokenizer /models/MiniCPM-o-4_5

# 3. Video-MME 全量精度
vllm bench serve \
  --omni --port 8091 --max-concurrency 4 \
  --dataset-name videomme --dataset-path /workspace/Video-MME \
  --num-prompts 2700 --trust-remote-code --no-oversample --disable-shuffle \
  --temperature 0 --output-len 128 \
  --videomme-pack-mode minicpm-frames --videomme-max-frames 96 --videomme-duration all \
  --model openbmb/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el \
  --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' \
  --tokenizer /models/MiniCPM-o-4_5
