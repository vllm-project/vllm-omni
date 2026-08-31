# 复现说明

## 1. 环境
- 镜像:quay.io/ascend/vllm-omni:v0.25.0-a3(官方赛题镜像)
- 硬件:Atlas 910C 单卡单 die
- 代码:本 fork 的 minicpm-challenge 分支(提交面见 docs/SUBMISSION.md)

## 2. 依赖
```bash
pip install stepaudio2-minicpmo
pip install step-audio2 --no-deps
pip install datasets==3.6.0 pyarrow fastparquet pytest==8.3.2 \
  pytest-asyncio==0.21.1 jiwer zhon gradio==6.24.0 funasr==1.4.2 \
  kaldiio torch-complex zhconv
```
WER 评测用的 paraformer-zh 与 wavlm-base-plus 为本地权重目录。

## 3. 数据准备
- Seed-TTS:seedtts_testset 解压至 /workspace/user_data/seedtts_testset
- Daily-Omni:qa.json + Videos/(全量 1197)
- Video-MME:videos 解压

## 4. 启动服务
所有优化已在代码默认值激活,直接用官方固定命令:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --stage-init-timeout 600 \
  --host 0.0.0.0 --port 8091
```

> 注意:请在 `/tmp` 等非源码目录下执行上面的 `vllm serve`。若在解压出的仓库根目录内启动,当前目录的 `vllm/` 会遮蔽 site-packages 里的 vllm 包导致 import 异常。

## 5. 评测命令

### 5.1 TTS 性能(官方 dfx 口径)
```bash
export HF_HUB_OFFLINE=1   # 离线环境必加：评估初始化会访问 HuggingFace，不通时卡死
vllm bench serve --omni --host 127.0.0.1 --port 8091 \
  --model openbmb/MiniCPM-o-4_5 \
  --tokenizer /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  --trust-remote-code --endpoint /v1/chat/completions --backend openai-chat-omni \
  --dataset-name seed-tts --dataset-path /workspace/user_data/seedtts_testset \
  --seed-tts-locale zh --num-prompts 32 --no-oversample --disable-shuffle --max-concurrency 1 \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}'
```
重启后首个 bench 先跑一轮热身再取正式读数。

### 5.2 TTS WER(全量 2020)
```bash
export SEED_TTS_EVAL_DEVICE=npu:0   # NPU 并发转写约 25 分钟;CPU 约 6 小时
export HF_HUB_OFFLINE=1   # 离线环境必加，否则评估初始化卡死
vllm bench serve --omni --port 8091 --max-concurrency 16 \
  --dataset-name seed-tts --dataset-path /workspace/user_data/seedtts_testset \
  --seed-tts-locale zh --num-prompts 2020 --disable-shuffle --no-oversample \
  --seed-tts-wer-eval --seed-tts-wer-save-items \
  --model openbmb/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}'
```

### 5.3 Daily-Omni(官方 file:// recipe)
```bash
vllm serve ... --interleave-mm-strings --allowed-local-media-path /workspace/Daily-Omni
vllm bench serve --omni --max-concurrency 10 --dataset-name daily-omni \
  --num-prompts 1197 --no-oversample --temperature 0 --output-len 512 \
  --daily-omni-input-mode all --daily-omni-pack-mode minicpm-interleave \
  --daily-omni-video-dir /workspace/Daily-Omni/Videos \
  --daily-omni-qa-json /workspace/Daily-Omni/qa.json \
  --model openbmb/MiniCPM-o-4_5 ...
```

### 5.4 Video-MME
全量 2700 条,96 帧无字幕,并发 4。

## 6. 预期结果
精度与性能数字见 docs/SUBMISSION.md。
