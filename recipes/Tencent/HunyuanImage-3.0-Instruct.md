# HunyuanImage-3.0-Instruct

> DiT-only text-to-image serving and benchmark with FP8, tensor parallelism,
> sequence parallelism, CFG parallelism, and ModelOpt mixed FP8/NVFP4
> checkpoints.

## Summary

- Vendor: Tencent Hunyuan
- Model: `tencent/HunyuanImage-3.0-Instruct`
- Task: Text-to-image generation
- Mode: Online serving and performance benchmarking, DiT stage only
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run or benchmark the HunyuanImage-3.0 DiT
stage directly. This is the recommended first setup when validating DiT
throughput, memory, FP8 kernels, sequence parallelism, or CFG parallelism.

The recipe covers three 4-GPU FP8 configurations and one 2-GPU ModelOpt mixed
FP8/NVFP4 configuration:

| Configuration | Parallelism | Notes |
| --- | --- | --- |
| `tp4_fp8` | TP=4 | Lowest per-GPU memory, higher communication overhead |
| `tp2_fp8_sp2` | TP=2, SP=2, Ulysses=2 | Splits sequence work across two GPUs per TP group |
| `tp2_fp8_cfgp2` | TP=2, CFG=2 | Runs CFG branches in parallel; fastest validated DiT setup |
| `tp2_mixed_fp8_nvfp4` | TP=2, EP enabled | Uses FP8 dense layers and NVFP4 routed experts on B200 |

## References

- Model: <https://huggingface.co/tencent/HunyuanImage-3.0-Instruct>
- Offline example:
  [`examples/offline_inference/hunyuan_image3`](../../examples/offline_inference/hunyuan_image3)
- Related PRs:
  [#2495](https://github.com/vllm-project/vllm-omni/pull/2495) for DiT performance CI,
  [#3055](https://github.com/vllm-project/vllm-omni/pull/3055) for GEBench accuracy CI,
  and [#3104](https://github.com/vllm-project/vllm-omni/pull/3104) for the T2I L3 dummy guard.

## Hardware Support

## GPU

### 4x H100/H800 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: CUDA-capable runtime matching the repository build
- vLLM minimum: 0.19.0, matching the current vLLM-Omni quickstart baseline.
- vLLM-Omni minimum: PR #2495 branch, or the first release that contains
  HunyuanImage-3.0 DiT serving with the CLI flags below.
- Optional environment variables:

```bash
export CACHE_DIT_VERSION=1.3.0
```

HunyuanImage-3.0 sets the diffusion attention backend to `TORCH_SDPA`
internally because the model mixes causal and full attention.

Graph mode is not part of this validated recipe. Keep `--enforce-eager` for
the FP8 DiT configurations below unless you separately validate graph mode for
the same checkpoint, parallelism, and image settings.

#### Commands

Start the DiT-only server with one of the following CLI-only configurations.
These commands use explicit CLI flags for all parallelism and runtime settings.

**TP=4 + FP8**

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --enable-diffusion-pipeline-profiler
```

**TP=2 + FP8 + SP=2**

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --tensor-parallel-size 2 \
  --usp 2 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --enable-diffusion-pipeline-profiler
```

**TP=2 + FP8 + CFG=2**

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --tensor-parallel-size 2 \
  --cfg-parallel-size 2 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --enable-diffusion-pipeline-profiler
```

Generate one 1024x1024 image:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A cinematic photo of a glass observatory on Mars at sunrise"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 5.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
     | cut -d',' -f2- \
     | base64 -d > hunyuan_image3_output.png
```

#### Benchmark

PR [#2495](https://github.com/vllm-project/vllm-omni/pull/2495) adds
performance CI configs for the same DiT-only settings. The CI step is
currently opt-in (gated by `RUN_HUNYUAN_IMAGE3_PERF=1`) with `soft_fail`
enabled, intended for initial data collection. Performance assertions are
skipped (`skip-performance-assertion: true`); the baseline values in the
JSON configs are reference-only and will be promoted to regression gates
once enough nightly data has been collected.

The user-facing equivalent is to launch one of the CLI commands above and
generate 1024x1024 images with 50 denoising steps.

#### Verification

Check that:

- The server responds on `http://localhost:8091/health`.
- The generation request writes a valid PNG file.
- Logs include `Selected CutlassFP8ScaledMMLinearKernel` for dense FP8
  linear layers and `Using TRITON Fp8 MoE` for MoE layers.
- With `--enable-diffusion-pipeline-profiler`, logs include per-stage timings
  such as `model.forward`, `patch_embed.forward`, `final_layer.forward`, and
  `vae.decode`.

Validated benchmark characteristics for 1024x1024, 50 denoising steps,
batch size 1:

| Configuration | Latency | Peak memory |
| --- | ---: | ---: |
| `tp4_fp8` | about 13.7s | about 47 GB |
| `tp2_fp8_sp2` | about 12.1s | about 66 GB |
| `tp2_fp8_cfgp2` | about 10.0s | about 66 GB |

#### Related Accuracy Smoke Data

PR [#3055](https://github.com/vllm-project/vllm-omni/pull/3055) adds a
DiT-only GEBench smoke setup for CI accuracy validation. Its validated
configuration was:

- Hardware: 4x H100.
- Runtime: TP=4 with expert parallel enabled, `bfloat16`,
  `distributed_executor_backend=mp`, `max_num_seqs=1`, `enforce_eager=True`.
- Task scope: T2I-only GEBench type3/type4, 4 samples per type, 28 denoising
  steps.
- Judge: `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`.

Validated score summary:

| Metric | Score |
| --- | ---: |
| overall_mean | 0.955 |
| type3 overall_mean | 0.91 |
| type4 overall_mean | 1.00 |

The CI assertion threshold is `overall_mean >= 0.45`, so the smoke result is
comfortably above the gate. The generate server and judge server run
sequentially through the `OmniServer` fixture, with GPU memory cleanup
between server lifetimes (for example via the `clean_gpu_memory_between_tests`
pytest fixture in the smoke path).

The lower-cost 2-GPU Instruct setup was tried for this smoke path but did not
fit in memory. A previous 2-GPU experiment used the base HunyuanImage-3.0
checkpoint with FP8, but that base checkpoint is not available in the CI HF
cache. The validated CI-ready Instruct setup is therefore 4x H100 TP=4 with
expert parallel.

#### Related Functional Guard

PR [#3104](https://github.com/vllm-project/vllm-omni/pull/3104) adds an L3
dummy guard for the T2I request path. The guard exercises
`HunyuanImage3Pipeline.forward()` without loading the full checkpoint by
stubbing `prepare_model_inputs()` and `_generate()`. It verifies propagation
of:

- prompt and system prompt selection;
- output image size;
- inference steps and guidance scale;
- request generator;
- image `DiffusionOutput` and `stage_durations`.

#### Notes

- This recipe is DiT-only and does not cover end-to-end HunyuanImage serving.
- `tp2_fp8_cfgp2` is usually fastest because CFG branches run in parallel.
  Individual layer timing can still look slower than `tp4_fp8` because each
  CFG branch uses TP=2, so each GPU owns a larger shard than in TP=4.
- `tp4_fp8` has the lowest per-GPU memory because weights are sharded across
  all four GPUs, but it pays more all-reduce communication overhead.
- `tp2_fp8_sp2` can improve model-forward latency by splitting sequence work,
  while adding all-to-all communication overhead.
- If you see OOM on 80GB GPUs, reduce image size, request concurrency, or use
  the TP=4 configuration before increasing batch size. GPU memory utilization
  is not a useful primary tuning knob for this DiT-only recipe.

### 2x B200 ModelOpt mixed FP8/NVFP4

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA environment with two B200 GPUs
- vLLM version: Match the vLLM-Omni checkout used for deployment
- vLLM-Omni version or commit: Use the commit that contains ModelOpt mixed
  FP8/NVFP4 checkpoint loading for diffusion models

#### Command

This configuration uses a pre-quantized ModelOpt mixed checkpoint for the
HunyuanImage-3.0 DiT stage. Dense attention and shared dense projections use
FP8, while routed MoE expert projections use NVFP4. Embeddings, norms,
routers, output layers, and non-DiT components stay in BF16.

Validated checkpoint:

```text
feizhai123/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8
```

Start the server with CUTLASS selected from the CLI for quantized linear and
MoE kernels:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve feizhai123/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8 \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --linear-backend cutlass \
  --moe-backend cutlass \
  --force-cutlass-fp8
```

#### Verification

Run a direct image generation request after the server is ready:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "feizhai123/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8",
    "messages": [
      {"role": "user", "content": "A cinematic portrait of an Adeptus Custodes warrior standing on Terra before the Imperial Palace, ornate golden armor, guardian spear, crimson cloak, marble steps, gothic architecture, holy golden light, highly detailed, no text, no watermark"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 20,
      "guidance_scale": 4.0,
      "true_cfg_scale": 4.0,
      "seed": 48
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
     | cut -d',' -f2- \
     | base64 -d > hunyuan_image3_mixed_nvfp4.png
```

Check that:

- The server responds on `http://localhost:8091/health`.
- The response contains one generated image.
- Logs show CUTLASS selection for ModelOpt FP8 dense layers and ModelOpt
  NVFP4 routed expert layers.

#### Benchmark

The numbers below are online HTTP measurements with 16 concurrent requests.
Each case serves 100 image-generation requests on the same 2-GPU B200 setup.

```text
Hardware: 2x B200
Tensor parallel size: 2
Requests: 100
Concurrency: 16
Resolution: 1024x1024
Denoising steps: 20
```

| Config | Mean | P90 | P99 | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 70.729s | 77.716s | 77.758s | 191,920 MiB |
| ModelOpt FP8 | 58.603s | 64.435s | 64.469s | 115,816 MiB |
| Mixed FP8/NVFP4 | 54.739s | 60.244s | 60.317s | 83,560 MiB |

#### Accuracy

Run the HunyuanImage-3.0 quantized DiT accuracy check with the BF16, FP8, and
mixed FP8/NVFP4 checkpoints:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
HUNYUAN_IMAGE3_QUANT_DEVICES=0,1 \
HUNYUAN_IMAGE3_QUANT_TP=2 \
HUNYUAN_IMAGE3_BF16_MODEL=/path/to/hunyuan-image3-bf16 \
HUNYUAN_IMAGE3_FP8_MODEL=/path/to/hunyuan-image3-modelopt-fp8 \
HUNYUAN_IMAGE3_NVFP4_MODEL=/path/to/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8 \
PYTHONPATH=/path/to/vllm-omni:/path/to/vllm:${PYTHONPATH:-} \
python -m pytest -s -v \
  tests/e2e/accuracy/test_hunyuan_image3.py \
  -k quantized_dit_matches_bf16_accuracy
```

The test compares quantized images with the BF16 image using CLIP-based image
and text scores, plus structural SSIM / PSNR checks. The current structural
thresholds are `SSIM >= 0.20` and `PSNR >= 10.0`.

#### Notes

- Use `--linear-backend cutlass` to select the validated quantized linear
  backend from the CLI.
- Use `--moe-backend cutlass` for the routed expert path.
- `--force-cutlass-fp8` keeps the FP8 dense layers on the CUTLASS FP8 path.
- The benchmark table is an online HTTP concurrency-16 result; do not compare
  it directly with loaded-once request-level benchmarks.
