# vLLM-Omni Nightly Common Error Signatures

Indexed by **cluster theme**; merge similar cases during triage to avoid repeating the same narrative per job.

## HF Gated Models (config/credentials)

```text
GatedRepoError: 403 Client Error
Cannot access gated repo ... black-forest-labs/FLUX.2-dev
Cannot access gated repo ... meta-llama/Meta-Llama-3.1-8B-Instruct
```

Typical jobs: Doc Test `more_cli_examples_004/005`; `test_flux_2_dev[*]`.

Verification: `huggingface-cli whoami`; confirm BK `HF_TOKEN` and model approval.

## Internal HF Mirror (infrastructure)

```text
HfHubHTTPError: 404 Not Found for url '...mirrors.../api/datasets/liarliar/Daily-Omni/...'
OSError: We couldn't connect to 'http://mirrors.tools.huawei.com/huggingface'
Error while downloading from http://mirrors.tools.huawei.com/huggingface/... timed out
```

Typical jobs: Omni Accuracy daily-omni; Qwen-Image-Edit function/perf; X2V Function weight download.

## Dataset Cache / Layout (config/permissions)

```text
PermissionError: [Errno 13] Permission denied: '/models/datasets/...*.lock'
FileNotFoundError: Seed-TTS meta not found: .../en/meta.lst
```

Typical jobs: `test_qwen3_omni_daily_omni_accuracy_bench`; `test_qwen3_omni_seed_tts_wer_bench`.

## SSL / Corporate Proxy (config/environment)

```text
SSLCertVerificationError ... self-signed certificate in certificate chain
proxyuk.huawei.com:8080
HTTPSConnectionPool(host='vllm-public-assets.s3...'
```

Typical jobs: Moss TTS setup (`raw.githubusercontent.com`); Diffusion Accuracy downloading `qwen-bear.png`; X2V Accuracy.

## Flashinfer JIT Missing Headers (compile/dependency)

```text
fatal error: cublasLt.h: No such file or directory
fatal error: nvrtc.h: No such file or directory
ninja ... fused_moe_90 ... exit status 1
```

Typical jobs: HunyuanImage3-DIT Accuracy.

## Hunyuan Deploy Config (test/config)

```text
ValueError: Pipeline 'hunyuan_image_3_moe' has async_chunk=True in deploy but no stage declares a next-stage input processor
```

## Accuracy Regression (test failure)

```text
AssertionError: PSNR below threshold for Qwen/Qwen-Image: got 29.xx, expected >= 30
```

## Qwen3-TTS Streaming (product/test)

```text
ValueError: Missing Qwen3-TTS ref context cache ... first chunk must include ref_code
Path load_format does not exist
```

Typical jobs: `test_voice_clone_streaming_001[async_chunk]`; `test_response_format_001[async_chunk]`.

## VoxCPM2 Audio Quality Threshold (test failure)

```text
AssertionError: Audio distortion detected: HNR=0.xx dB < 1.0 dB
```

## Omni Doc Examples (test failure)

```text
AssertionError: The output does not contain any of the keywords
```

## Omni Online Multimodal (test failure; correlate with test name)

```text
test_text_audio_to_text_audio_001
test_audio_to_audio
```

Logs often show only `FAILED` with no traceback (custom runner); expand context around the test case or the pytest `--tb=short` section.

## Diffusion Worker Runtime Crash (infrastructure/product)

```text
Diffusion worker(s) died unexpectedly
RuntimeError: Orchestrator died unexpectedly
```

May appear in perf jobs but summary still passed (intermittent); mark failed if bench was interrupted.

## Quantization Worker EOF (infrastructure)

```text
EOFError
... multiproc ... _recv
FAILED tests/diffusion/quantization/test_quantization_fp8.py::
```

## Perf Bench Request Failure (timeout/performance)

```text
httpx.ReadTimeout: timed out
AssertionError: Request failures exist
Benchmark result file not generated
```
