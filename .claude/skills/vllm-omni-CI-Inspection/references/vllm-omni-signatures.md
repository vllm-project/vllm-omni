# vLLM-Omni Nightly 常见错误签名

按 **聚类主题** 索引；分诊时合并同类，避免每个 job 重复叙述。

## HF Gated 模型（配置/凭据）

```text
GatedRepoError: 403 Client Error
Cannot access gated repo ... black-forest-labs/FLUX.2-dev
Cannot access gated repo ... meta-llama/Meta-Llama-3.1-8B-Instruct
```

典型 job：Doc Test `more_cli_examples_004/005`；`test_flux_2_dev[*]`。

验证：`huggingface-cli whoami`；确认 BK `HF_TOKEN` 与模型审批。

## 内网 HF 镜像（基础设施）

```text
HfHubHTTPError: 404 Not Found for url '...mirrors.../api/datasets/liarliar/Daily-Omni/...'
OSError: We couldn't connect to 'http://mirrors.tools.huawei.com/huggingface'
Error while downloading from http://mirrors.tools.huawei.com/huggingface/... timed out
```

典型 job：Omni Accuracy daily-omni；Qwen-Image-Edit function/perf；X2V Function 权重下载。

## 数据集缓存 / 布局（配置/权限）

```text
PermissionError: [Errno 13] Permission denied: '/models/datasets/...*.lock'
FileNotFoundError: Seed-TTS meta not found: .../en/meta.lst
```

典型 job：`test_qwen3_omni_daily_omni_accuracy_bench`；`test_qwen3_omni_seed_tts_wer_bench`。

## SSL / 企业代理（配置/环境）

```text
SSLCertVerificationError ... self-signed certificate in certificate chain
proxyuk.huawei.com:8080
HTTPSConnectionPool(host='vllm-public-assets.s3...'
```

典型 job：Moss TTS setup（`raw.githubusercontent.com`）；Diffusion Accuracy 下载 `qwen-bear.png`；X2V Accuracy。

## Flashinfer JIT 缺头文件（编译/依赖）

```text
fatal error: cublasLt.h: No such file or directory
fatal error: nvrtc.h: No such file or directory
ninja ... fused_moe_90 ... exit status 1
```

典型 job：HunyuanImage3-DIT Accuracy。

## Hunyuan deploy 配置（测试/配置）

```text
ValueError: Pipeline 'hunyuan_image_3_moe' has async_chunk=True in deploy but no stage declares a next-stage input processor
```

## 精度回归（测试失败）

```text
AssertionError: PSNR below threshold for Qwen/Qwen-Image: got 29.xx, expected >= 30
```

## Qwen3-TTS 流式（产品/测试）

```text
ValueError: Missing Qwen3-TTS ref context cache ... first chunk must include ref_code
Path load_format does not exist
```

典型 job：`test_voice_clone_streaming_001[async_chunk]`；`test_response_format_001[async_chunk]`。

## VoxCPM2 音质门限（测试失败）

```text
AssertionError: Audio distortion detected: HNR=0.xx dB < 1.0 dB
```

## Omni Doc 示例（测试失败）

```text
AssertionError: The output does not contain any of the keywords
```

## Omni 在线多模态（测试失败，需结合用例名）

```text
test_text_audio_to_text_audio_001
test_audio_to_audio
```

日志常仅 `FAILED` 无 traceback（自定义 runner）；需扩大该用例前后文或 pytest `--tb=short` 段。

## Diffusion worker 运行时崩溃（基础设施/产品）

```text
Diffusion worker(s) died unexpectedly
RuntimeError: Orchestrator died unexpectedly
```

perf job 中可能出现但 summary 仍 passed（偶发）；若 bench 中断则记 failed。

## Quantization worker EOF（基础设施）

```text
EOFError
... multiproc ... _recv
FAILED tests/diffusion/quantization/test_quantization_fp8.py::
```

## Perf bench 请求失败（超时/性能）

```text
httpx.ReadTimeout: timed out
AssertionError: Request failures exist
Benchmark result file not generated
```
