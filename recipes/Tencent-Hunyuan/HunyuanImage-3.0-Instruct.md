# HunyuanImage-3.0-Instruct DiT Image Generation on 4x GPU

> DiT-only text-to-image recipe for HunyuanImage-3.0-Instruct with FP8,
> tensor parallelism, sequence parallelism, and CFG parallelism.

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

The recipe covers three 4-GPU configurations:

| Configuration | Parallelism | Notes |
| --- | --- | --- |
| `tp4_fp8` | TP=4 | Lowest per-GPU memory, higher communication overhead |
| `tp2_fp8_sp2` | TP=2, SP=2, Ulysses=2 | Splits sequence work across two GPUs per TP group |
| `tp2_fp8_cfgp2` | TP=2, CFG=2 | Runs CFG branches in parallel; fastest validated DiT setup |

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
