# 复现说明

## 1. 环境准备
- 镜像：quay.io/ascend/vllm-omni:v0.25.0-a3
- 硬件：Atlas A3（910C）单卡
- 代码分支：minicpm-challenge

## 2. 安装依赖
pip install stepaudio2-minicpmo
pip install step-audio2 --no-deps
pip install datasets==3.6.0 pyarrow fastparquet pytest==8.3.2 pytest-asyncio==0.21.1 jiwer zhon gradio==6.24.0 funasr==1.4.2 kaldiio torch-complex zhconv

## 3. 数据准备
- Seed-TTS：解压 seedtts_testset.tar 到 /workspace/seedtts_testset
- Daily-Omni：转换生成 qa.json 与 Videos/
- Video-MME：复制 videos_chunked_*.zip 并解压到 /workspace/Video-MME/videos

## 4. 启动服务
注意：跑 Daily-Omni 时需追加 --interleave-mm-strings --allowed-local-media-path /workspace/Daily-Omni；跑 Video-MME 时追加 --allowed-local-media-path /workspace/Video-MME；跑 Seed-TTS 无需追加。
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /models/MiniCPM-o-4_5 --omni --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --stage-init-timeout 600 --host 0.0.0.0 --port 8091

## 5. 执行评测
按 run_benchmarks.sh 顺序执行三项 Benchmark。

## 6. 预期结果
精度四项全部通过，性能 RTF 0.36 / TTFT 372ms / TTFP 1105ms。

## 7. 依赖修补说明

官方镜像预装依赖与实际评测存在差异，本次实测发现并修补如下：
- 镜像声称预装 fastparquet，实际缺失，需手动安装
- step-audio2 必须使用 --no-deps 安装，避免覆盖镜像内 torch/torch_npu
- pytest 版本需回滚到 8.3.2，否则与镜像内 pytest-asyncio 不兼容
- funasr 需搭配 kaldiio 和 torch-complex 才能正常计算 WER
