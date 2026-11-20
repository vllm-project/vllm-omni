# HunyuanImage-3.0-Instruct

> DiT-only text-to-image serving and benchmark with FP8, tensor parallelism,
> sequence parallelism, CFG parallelism, and ModelOpt mixed FP8/NVFP4
> checkpoints.

## Summary

- Vendor: Tencent Hunyuan
- Model: `tencent/HunyuanImage-3.0-Instruct`
- Task: Text-to-image generation, image editing; text-to-text and image-to-text understanding
- Mode: Offline understanding, online serving, and performance benchmarking
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
- Shared T2I example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Shared IT2I example:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Shared T2T/I2T example:
  [`examples/offline_inference/x_to_text/x_to_text.py`](../../examples/offline_inference/x_to_text/x_to_text.py)
- Related PRs:
  [#2495](https://github.com/vllm-project/vllm-omni/pull/2495) for DiT performance CI,
  [#3055](https://github.com/vllm-project/vllm-omni/pull/3055) for GEBench accuracy CI,
  and [#3104](https://github.com/vllm-project/vllm-omni/pull/3104) for the T2I L3 dummy guard.

## Hardware Support

## GPU

### Shared T2T/I2T offline example

For text-to-text, the shared x-to-text example automatically selects the
AR-only Hunyuan deploy config and applies the checkpoint's prompt tokens and
stop-token rules:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompt "Explain why the sky appears blue."
```

Add `--image` for image-to-text:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --image /path/to/input.jpg \
  --prompt "Describe the content of this image."
```

The AR-only default uses four GPUs. Pass `--deploy-config` to override the
layout.

### AR Model Runner V2

HunyuanImage3 AR follows the repository's Model Runner V2 default, including
the AR-only, AR+DiT and Mooncake KV connector deployments and accuracy tests.

The AR sampler preserves comprehension token blocking, image-generation stage
transitions, and greedy aspect-ratio selection using MRv2's device-side
token history. Speculative decoding must be disabled for this stage.

The adapter uses the existing generic Omni model states and does not change
attention. NPU execution requires an Ascend runtime matching the repository's
vLLM version; the retained v0.33 NPU image is not qualified for vLLM 0.34.

### Shared T2I/IT2I offline examples

Text-to-image and image-editing route through the shared task examples, with
all HunyuanImage-3.0-specific knobs (`bot_task`, `use_system_prompt`,
`system_prompt`, `negative_prompt`) declared in
`vllm_omni/model_extras/hunyuan_image3.py` and passed via `--extra-body` /
`--extra-args`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_moe.yaml \
  --trust-remote-code \
  --prompt "A cute cat sitting on a windowsill watching the sunset" \
  --height 1024 --width 1024 \
  --guidance-scale 5.0 --num-inference-steps 50 --seed 42 \
  --extra-body '{"bot_task": "think", "use_system_prompt": "en_recaption"}' \
  --output hunyuan_t2i.png
```

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_moe.yaml \
  --trust-remote-code \
  --image /path/to/image.png \
  --prompt "Make the petals neon pink" \
  --guidance-scale 5.0 --num-inference-steps 50 --seed 42 \
  --extra-args '{"bot_task": "think"}' \
  --output hunyuan_edit.png
```

`image_edit.py` accepts up to 3 reference images (repeat `--image`) for
HunyuanImage-3.0's multi-image fusion. All four shared-script paths (t2i,
it2i, i2t, t2t) build the AR prefill and stop tokens through the same
declarative seam (`build_ar_stage_inputs` / `build_x_to_text_prompt`, both
wrapping `prompt_utils.build_ar_prompt_inputs`), so prompt formatting is
identical across them. The OpenAI server's `serving_chat.py` does not go
through this seam yet -- see that function's docstring for the resulting
`bot_task`-resolution divergence risk when omitted from `extra_body`.

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
export CACHE_DIT_VERSION=1.5.0
```

HunyuanImage-3.0 sets the diffusion attention backend to `TORCH_SDPA`
internally because the model mixes causal and full attention.

Graph mode is not part of this validated recipe. Keep `--enforce-eager` for
the FP8 DiT configurations below unless you separately validate graph mode for
the same checkpoint, parallelism, and image settings.

#### Commands

Each command selects the DiT-only deploy, which defines one diffusion stage
on visible devices `0,1,2,3`. Each parallelism configuration below uses all
four devices.

##### TP=4 + FP8

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --enable-diffusion-pipeline-profiler
```

##### TP=2 + FP8 + SP=2

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --tensor-parallel-size 2 \
  --usp 2 \
  --quantization fp8 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --enable-diffusion-pipeline-profiler
```

##### TP=2 + FP8 + CFG=2

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml \
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
enabled, intended for initial data collection. Perf runners no longer
compare metrics against JSON baselines at assert time; the `baseline`
blocks in the configs are reference-only and are copied into result JSON
for reporting. Regression gates can be reintroduced once enough nightly
data has been collected.

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

### Scheduler-managed paged KV (request mode)

HunyuanImage-3 can use Scheduler-managed paged KV in request-level execution.
The current paged implementation uses unquantized BF16 KV and requires the
logical `FLASH_ATTN` backend. The following command reuses the standard DiT
deploy file and supplies the paged-only stage settings with
`--stage-overrides`:

For the mode-selection guide, configuration field reference, and parallelism
limits, see the [Scheduler-Managed Paged KV Cache user guide](../../docs/user_guide/diffusion/paged_kv_cache.md).

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --trust-remote-code \
  --port 8091 \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --stage-overrides \
  '{"0":{"diffusion_kv_mode":"paged_scheduler","diffusion_kv_max_rows_per_request":2,"kv_cache_memory_bytes":536870912,"step_execution":false,"diffusion_attention_backend":"FLASH_ATTN"}}'
```

Here `kv_cache_memory_bytes` reserves a 512 MiB physical KV pool per Worker
and rank, and `diffusion_kv_max_rows_per_request=2` reserves the two internal
rows used by standard positive/negative CFG. Use `1` when CFG is disabled.
Keep `max_num_seqs=1`: this paged Hunyuan path does not currently batch
independent public requests.

Generate an image after `/health` reports ready:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "modalities": ["image"],
    "messages": [{"role": "user", "content": "A red ceramic vase on a wooden table"}],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 20,
      "guidance_scale": 5.0,
      "seed": 1234
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
    | cut -d',' -f2- | base64 -d > hunyuan_image3_paged.png
```

This mode is also implemented on Ascend NPU, where `FLASH_ATTN` resolves to
Ascend FIA. Hunyuan paged step execution, quantized paged KV, and CPU/DLO
offload combinations are not validated; use `dense_legacy` for those cases.

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

Select the DiT-only deployment and give stage 0 both visible devices. The
quantized linear and MoE kernels use CUTLASS.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve feizhai123/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8 \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --stage-overrides '{"0":{"devices":"0,1"}}' \
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

The CI accuracy matrix below replaces the former opt-in BF16 versus
FP8/NVFP4 comparison tests.

#### Notes

- Use `--linear-backend cutlass` to select the validated quantized linear
  backend from the CLI.
- Use `--moe-backend cutlass` for the routed expert path.
- `--force-cutlass-fp8` keeps the FP8 dense layers on the CUTLASS FP8 path.
- The benchmark table is an online HTTP concurrency-16 result; do not compare
  it directly with loaded-once request-level benchmarks.

## CI performance and accuracy matrix

The test matrix includes these four-card workloads. AR+DiT accuracy is
marked `local_model` and excluded from CUDA nightly `full_model` selection;
its BF16 TP=2 DiT stage exceeded 80 GB per GPU during model initialization.
The NPU accuracy job explicitly selects `(full_model or local_model)` to
run all three accuracy cases. Other workloads use `full_model` selection:

| Workload | Weights | Parallelism | Dataset / output | Concurrency | Metrics |
| --- | --- | --- | --- | --- | --- |
| AR + DiT performance | Distil, FP8 | AR TP=2 (0,1), DiT TP=1/SP=2 (2,3) | custom IT2I | 1,16 | latency, throughput |
| DiT performance | Distil, FP8 | TP=1/SP=4 (0,1,2,3) | random T2I | 1,4 | latency, throughput |
| DiT accuracy | Distil, BF16 | TP=4 (0,1,2,3) | IT2I, one output image | 1 | PSNR, SSIM |
| AR + DiT accuracy | Distil, BF16 | AR TP=2 (0,1), DiT TP=2 (2,3) | IT2I, one output image | 1 | PSNR, SSIM, CoT similarity, image similarity, CoT prefix |

Performance uses `tencent/HunyuanImage-3.0-Instruct-Distil` with
`--quantization fp8` for both stages. It does not benchmark the original
Instruct checkpoint with an eight-step schedule. SP uses Ulysses. Both
performance deploys set `async_chunk: false` (non-duplex).

Performance cases reuse the production `hunyuan_image_3_moe.yaml` and
`hunyuan_image3_dit.yaml` deployments. Their JSON `stage_overrides` selects
the four-card topology on CUDA and NPU, uses the shared-memory connector
for colocated AR+DiT, and limits AR memory utilization to 0.85 to leave
headroom for IT2I VAE encoding. No performance-only deployment YAML is needed.

The AR+DiT workload extends the existing
`tests/dfx/perf/tests/test_hunyuan_image3_ar_tp2_dit_sp2.json`: it retains the custom
prompt, input image, `/v1/images/edits` endpoint and `think_recaption` task.
The DiT-only request format can also be compared with
`tests/dfx/perf/tests/test_hunyuan_image3_prefix_caching.json`, which uses
the two-image dataset shared by the prefix-cache accuracy tests. This
existing local-model performance suite is retained alongside the new matrix;
it does not supply an accuracy reference image. The previous
AR+DiT H200 performance baseline describes Instruct with DiT TP=2, so it
does not calibrate the distilled FP8 TP=1/SP=2 workload.

Run from the repository root on four H100/B200 GPUs with the matching
vLLM/vLLM-Omni runtime, cached weights and access to the input image URLs:

```bash
pytest -sv tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_hunyuan_image3_ar_tp2_dit_sp2.json -m "full_model and diffusion and H100 and cards_4" --run-level full_model
pytest -sv tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_hunyuan_image3_dit_sp4.json -m "full_model and diffusion and H100 and cards_4" --run-level full_model
pytest -sv tests/e2e/accuracy/test_hunyuan_image3.py -k test_hunyuan_image3_ar_tp2_dit_tp2_it2i -m "local_model and diffusion and H100 and cards_4" --run-level full_model
```

Both accuracy workloads use the distilled checkpoint in BF16 without
quantization and with eight inference steps. Performance workloads retain FP8. AR+DiT accuracy also needs
`FlagEmbedding` and the embedding/CLIP model caches. `HUNYUAN_MODEL_PATH`
(AR+DiT) and `HUNYUAN_IMAGE3_MODEL` (DiT-only) may point to local distilled
checkpoints. AR+DiT retains its existing distilled image/CoT references and
thresholds; the BF16 outputs must still pass those checks.

DiT-only IT2I uses `tests/assets/hunyuan_image3/it2i.jsonl`, both input
images in their listed order, `bot_task=none`, `sys_type=en_unified`, seed
42, 8 steps, guidance 2.5, BF16 without quantization, and one 1280x720 output.
It selects the official distilled direct-DiT reference for the current
platform from `tests/assets/hunyuan_image3/`:

| Platform profile | Reference PNG | SSIM floor | PSNR floor (dB) |
| --- | --- | --- | --- |
| CUDA: H200/H100/B200/default | `hunyuan_image3_dit_distill_it2i_ref.png` | 0.95 | 29.0 |
| NPU: A3 | `hunyuan_image3_dit_distill_it2i_ref_npu.png` | 0.92 | 28.0 |

These are acceptance floors, with margin below the five-run calibration
minima, rather than the measured means. The CUDA measurements were collected
on H200; H100/B200 and the default profile use the same configured floors.
See the [CUDA/H200 calibration](https://github.com/vllm-project/vllm-omni/pull/10408#issuecomment-6940430119)
and [A3 calibration](https://github.com/vllm-project/vllm-omni/pull/10408#issuecomment-6936673053).
The AR+DiT image/CoT references and thresholds remain separate. No reference
environment variables are required:

```bash
pytest -sv tests/e2e/accuracy/test_hunyuan_image3.py::test_hunyuan_image3_dit_tp4_it2i_online --run-level full_model
pytest -sv tests/e2e/accuracy/test_hunyuan_image3.py::test_hunyuan_image3_dit_tp4_it2i_online -m "full_model and diffusion and H100 and cards_4" --run-level full_model
```

The two new performance workloads intentionally have empty `baseline` maps
until calibration on the target CI hardware. The unified runner records `mean_e2el_ms`,
`p99_e2el_ms` and `request_throughput`, but empty baselines do not establish a
performance regression gate. Populate each hardware bucket with measured
values before treating the matrix as calibrated. Sweep baseline values are
lists in concurrency order: `[1,16]` for AR+DiT and `[1,4]` for DiT. Do not
reuse the previous Instruct TP=2/SP=2 or TP=4 benchmark numbers.
