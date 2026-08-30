# 复现说明

## 1. 环境准备

- 镜像：quay.io/ascend/vllm-omni:v0.25.0-a3
- 硬件：Atlas A3（910C）单卡
- 代码分支：minicpm-challenge
- 当前优化：token2wav_n_timesteps=3

## 2. 安装依赖

pip install stepaudio2-minicpmo
pip install step-audio2 --no-deps
pip install datasets==3.6.0 pyarrow fastparquet
pip install pytest==8.3.2 pytest-asyncio==0.21.1
pip install jiwer zhon gradio==6.24.0
pip install funasr==1.4.2 kaldiio torch-complex zhconv

## 3. 数据准备

### Seed-TTS

mkdir -p /workspace/seed-tts-eval
tar -xf /workspace/shared_assets/datasets/CowboyZ/seed-tts-eval/seedtts_testset.tar -C /workspace/seed-tts-eval

### Daily-Omni

需要包含：
- /workspace/Daily-Omni/qa.json
- /workspace/Daily-Omni/Videos/视频目录

### Video-MME

需要包含：
- /workspace/Video-MME/test-00000-of-00001.parquet
- /workspace/Video-MME/videos/视频文件

## 4. 启动服务

跑 Seed-TTS 时：

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /workspace/local_models/OpenBMB/MiniCPM-o-4_5 --omni --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --stage-init-timeout 600 --host 0.0.0.0 --port 8091

跑 Daily-Omni 时需追加：
--interleave-mm-strings --allowed-local-media-path /workspace/Daily-Omni

跑 Video-MME 时需追加：
--allowed-local-media-path /workspace/Video-MME

## 5. 执行评测

### Seed-TTS 全量

SEED_TTS_SIM_EVAL=1 SEED_TTS_WER_EVAL=1 vllm bench serve --omni --port 8091 --trust-remote-code --max-concurrency 1 --num-warmups 2 --dataset-name seed-tts --dataset-path /workspace/seed-tts-eval --seed-tts-locale zh --num-prompts 2020 --disable-shuffle --no-oversample --seed-tts-wer-eval --model openbmb/MiniCPM-o-4_5 --endpoint /v1/chat/completions --backend openai-chat-omni --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' --tokenizer /workspace/local_models/OpenBMB/MiniCPM-o-4_5

### Daily-Omni 全量

vllm bench serve --omni --port 8091 --max-concurrency 10 --dataset-name daily-omni --num-prompts 1197 --trust-remote-code --no-oversample --temperature 0 --output-len 512 --daily-omni-input-mode all --daily-omni-pack-mode minicpm-interleave --daily-omni-video-dir /workspace/Daily-Omni/Videos --daily-omni-qa-json /workspace/Daily-Omni/qa.json --model openbmb/MiniCPM-o-4_5 --endpoint /v1/chat/completions --backend openai-chat-omni --percentile-metrics ttft,tpot,itl,e2el --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' --tokenizer /workspace/local_models/OpenBMB/MiniCPM-o-4_5

### Video-MME 全量

vllm bench serve --omni --port 8091 --max-concurrency 4 --dataset-name videomme --dataset-path /workspace/Video-MME --num-prompts 2700 --trust-remote-code --no-oversample --disable-shuffle --temperature 0 --output-len 128 --videomme-pack-mode minicpm-frames --videomme-max-frames 96 --videomme-duration all --model openbmb/MiniCPM-o-4_5 --endpoint /v1/chat/completions --backend openai-chat-omni --percentile-metrics ttft,tpot,itl,e2el --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' --tokenizer /workspace/local_models/OpenBMB/MiniCPM-o-4_5

## 6. 预期结果

| Benchmark | 指标 | 结果 |
|-----------|------|------|
| Seed-TTS | WER | 1.37% |
| Seed-TTS | SIM | 0.8459 |
| Seed-TTS | RTF | 0.30 |
| Daily-Omni | Accuracy | 78.18% |
| Video-MME | Accuracy | 69.59% |

## 7. 依赖修补说明

- step-audio2 必须使用 --no-deps 安装
- fastparquet 镜像未预装，需手动安装
- pytest 需回滚到 8.3.2
- funasr 需搭配 kaldiio 和 torch-complex 才能正常计算 WER
