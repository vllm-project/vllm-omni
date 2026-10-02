# MammothModa2

> MammothModa2-Preview and MammothModa2-Dev unified understanding and generation

## Summary

- Vendor: ByteDance Research
- Models: `bytedance-research/MammothModa2-Preview`, `bytedance-research/MammothModa2-Dev`
- Tasks: Preview and Dev text-to-image (AR → DiT); Dev text/image understanding
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to run MammothModa2-Preview through shared task-oriented
offline examples. Text-to-image uses the shared image example (`text_to_image.py`) instead of a model-specific script.
The generic example formats the AR prompt, drives the AR → DiT stage pipeline,
and forwards MammothModa2-specific generation parameters through the
pipeline-declared `extra_body` contract.

MammothModa2's DiT stage runs in the shared diffusion runtime in request mode.
The first integration intentionally supports one request and one image per
forward only (`max_num_seqs: 1`, `num_outputs_per_prompt: 1`). Request-level
batching, step execution, continuous batching, cache acceleration,
compilation, quantization, parallelism, and offload are not enabled by this
recipe.

Image size, seed, guidance, and denoising steps use the standard diffusion
request fields. `cfg_range` remains a MammothModa2-specific `extra_body`
parameter. For compatibility, the runtime also accepts the former
`text_guidance_scale` and `num_inference_steps` keys in `extra_body`; when
present and non-null, those keys take precedence over the standard fields.

## References

- Upstream model:
  [`bytedance-research/MammothModa2-Preview`](https://huggingface.co/bytedance-research/MammothModa2-Preview)
- Dev model:
  [`bytedance-research/MammothModa2-Dev`](https://huggingface.co/bytedance-research/MammothModa2-Dev)
- Related offline example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Related T2T/I2T example:
  [`examples/offline_inference/x_to_text/x_to_text.py`](../../examples/offline_inference/x_to_text/x_to_text.py)
- Declared parameters:
  [`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)
- Deploy config:
  [`vllm_omni/deploy/mammoth_moda2.yaml`](../../vllm_omni/deploy/mammoth_moda2.yaml)

## Hardware Support

The default deploy config places both the AR and DiT stages on one GPU
(`devices: "0"`). Its committed `gpu_memory_utilization` split is 0.5 for
stage 0 and 0.3 for stage 1. The A800 validation section below also shows a
two-GPU placement with one stage per GPU for attributable timing and memory;
the measured results are summarized below.

## GPU

### Optional FP8 AR KV cache

For CUDA deployments, `mammoth_moda2_fp8_kv.yaml` is an opt-in preset that
keeps the Stage 0 AR KV cache of decoder layer 0 in BF16 and stores the other
27 layers as FP8 E4M3 (`kv_cache_dtype_skip_layers: ["0"]`; write the layer
indices as quoted strings). Stage 1 remains on
`kv_cache_dtype=auto`; its DiT execution is unaffected. This setting quantizes
only the autoregressive KV cache. It is neither FP8 weight/activation
quantization nor vLLM-Omni diffusion KV-cache quantization.

Use the preset in place of the default deploy config:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2_fp8_kv.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
  --output mammoth_t2i.png
```

The preset was first validated on one NVIDIA H800 80GB with CUDA and
FlashAttention 3, with every layer in FP8. The native and FP8 runs used the
same model, code revision, and downstream configuration.

| Metric | Native BF16 (`kv_cache_dtype=auto`) | FP8 E4M3 (`fp8_e4m3`) | Change |
| --- | ---: | ---: | ---: |
| KV cache memory | 15.59 GiB | 15.55 GiB | -0.04 GiB |
| GPU KV cache size | 145,904 tokens | 291,232 tokens | +99.6% |
| Maximum concurrency at 8,192 tokens | 17.81x | 35.55x | +99.6% |
| Steady-state AR median | 71.191 s | 79.816 s | +12.1% |
| Steady-state end-to-end median | 83.840 s | 92.473 s | +10.3% |
| Steady-state DiT median | 12.574 s | 12.561 s | effectively unchanged |

The FP8 run used `kv_cache_dtype=fp8_e4m3` only for Stage 0; Stage 1 used
`kv_cache_dtype=auto`. The nearly unchanged reserved cache memory holds almost
twice as many tokens because FP8 reduces the bytes per cached token. This is a
capacity/concurrency tradeoff: the measured AR and end-to-end latencies were
higher than the native-BF16 baseline.

On H800, a 1024x1024 fixed-seed smoke test with every layer in FP8 completed
successfully with no obvious visual failure. FP8 is lossy, so numerical or
image-quality equivalence with BF16 is not implied.

On one A800 80GB (vLLM 0.30.0), FP8 KV layers run on FlashInfer and BF16 layers
on FlashAttention 2. Text-to-image at 1024x1024, 50 steps, guidance 4.0, seeds 42
and 1-5, with a studio tabby cat prompt and a peephole-view Samoyed prompt;
an image counts when it shows the prompted subject and scene.

| Stage 0 KV cache | GPU KV cache size | Cat images that follow the prompt | Samoyed images that follow the prompt |
| --- | ---: | ---: | ---: |
| BF16 on all 28 layers (`auto`) | 147,408 tokens | 6/6 | 6/6 |
| FP8 E4M3 on all 28 layers | 294,816 tokens | 1/6 | 0/6 |
| Layer 0 BF16, other 27 layers FP8 E4M3 (this preset) | 284,640 tokens | 6/6 | 6/6 |

The KV cache sizes above still count the 28 attention layers of the replaced
Qwen-VL language model, which #8095 removes: with it, BF16 goes from 147,408 to
294,816 tokens and all-FP8 from 294,816 to 589,632. The prompt-following
columns do not depend on it.

With all 28 layers in FP8, most cat images become a framed print on a wall.
Keeping layer 0 in BF16 restores them; keeping only layer 27, which has the
largest key magnitude, does not. The H800 numbers above were measured with all 28
layers in FP8; this preset has been run on A800 only.

### 1x L40S 48GB

> **48 GB config adjustment:** the committed
> `vllm_omni/deploy/mammoth_moda2.yaml` uses
> `gpu_memory_utilization` 0.5 / 0.3 (sized for ~80 GB). To fit on a 48 GB L40S,
> set the stage-0 (AR) value to `0.8` and the stage-1 (DiT) value to `0.16`
> before running. (On an ~80 GB GPU, leave the defaults unchanged.)

### 1x NVIDIA A800 80GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with one A800 80 GB
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Offline Commands

Download the model:

```bash
hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

Run text-to-image with the shared offline example from the repository root. The
deploy config sets `trust_remote_code`, so no extra flag is needed:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --guidance-scale 4.0 \
  --num-inference-steps 50 \
  --extra-body '{"cfg_range": [0.0, 1.0]}' \
  --output mammoth_t2i.png
```

The standard diffusion request fields are `height`, `width`, `seed`,
`guidance_scale`, and `num_inference_steps`; use their corresponding CLI flags
shown above. `--height` and `--width` must be multiples of 16.

`cfg_range` is the only recommended MammothModa2 field in `--extra-body`; it
sets the relative step range `[start, end]` over which CFG is applied (default
`[0.0, 1.0]`). For compatibility, `text_guidance_scale` and
`num_inference_steps` remain accepted `extra_body` aliases and, when non-null,
take precedence over the standard request fields. Model extras are filtered
against the declared `extra_body_params` (see
[`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)),
so unknown MammothModa2 extras may be dropped.

Run text-to-text through the shared understanding example. It recognizes the
MammothModa2 checkpoint and automatically selects `mammoth_moda2_ar.yaml`:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Preview \
  --prompt "Explain multimodal generation in three sentences."
```

Add an image for image-to-text or image summarization. The shared example
uses MammothModa2's chat and vision-token template:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Preview \
  --image /path/to/input.jpg \
  --prompt "Please summarize the content of this image."
```

#### Verification

The example writes the generated image to the `--output` path. Confirm the file
exists and is a valid image:

```bash
ls -lh mammoth_t2i.png
python -c "from PIL import Image; print(Image.open('mammoth_t2i.png').size)"
```

### 2x NVIDIA A800 80GB validation

Use one A800 per stage so AR and DiT memory and timing are attributable. The
per-stage override changes placement only; both stages remain single-rank.

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ./MammothModa2-Preview --omni \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --stage-overrides '{"0":{"devices":"0"},"1":{"devices":"1"}}' \
  --port 8099 \
  --log-stats
```

Startup logs should identify stage 1 as `StageDiffusionClient` and resolve it
to `MammothModa2DiTPipeline`. `DiffusionEngine` step timing is a DEBUG-level,
per-request message, so it appears only after sending a text-to-image request
with `VLLM_LOGGING_LEVEL=DEBUG`; it is not a startup marker. Seeing the legacy
generation model runner for stage 1 is a failed migration.

#### Migration benchmark

The request-mode migration was checked on 2x NVIDIA A800 80GB PCIe with AR on
GPU 0 and DiT on GPU 1. Each revision ran one warmup followed by 10 serial
measured requests in the same initialized process. Both used BF16 eager mode,
1024x1024 output, 50 denoising steps, guidance scale 4.0, seed 42, and no
diffusion cache. The baseline was the pre-migration revision `caed3061`; the
candidate was `19de562a`. Lower latency is better.

| Metric | Baseline p50 | Baseline p95 | Candidate p50 | Candidate p95 |
| --- | ---: | ---: | ---: | ---: |
| End-to-end latency | 105.36 s | 106.01 s | 104.94 s | 105.79 s |
| AR stage latency | 86.97 s | 87.61 s | 86.26 s | 87.11 s |
| DiT stage latency | 18.27 s | 18.41 s | 18.60 s | 18.62 s |

Peak sampled device memory was 39,209 MiB on the AR GPU for both revisions.
The DiT GPU used 11,089 MiB for the baseline and 10,967 MiB for the candidate.
The candidate's shared runtime reported 372.02 ms p50 per denoising step and a
5.89 ms p50 AR-to-diffusion adapter time. All measured requests completed and
both revisions produced valid, prompt-aligned 1024x1024 RGB images. The small
latency differences are regression evidence, not a statistically significant
speedup claim.

### 1x AMD MI300X, MammothModa2 Preview (pre-migration baseline)

#### Environment

- OS: Linux 6.8.0-134-generic, x86_64
- Container: official ROCm image built from `docker/Dockerfile.rocm`
- Python: 3.12.13
- PyTorch: 2.11.0+gitd0c8b1f
- Driver / runtime: AMD 6.19.14.31400000 / ROCm 7.2.53211
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 191.69 GiB visible HBM
- vLLM version: 0.27.0+rocm723
- vLLM Omni version or commit: `73e1368c7bb940efe1a025859c9d6c8eeeb2e3f0`
- Installed vLLM Omni package metadata: `0.27.0rc2.dev44+g55abdade9.rocm`

#### Offline Commands

The checked run used the committed stage split, with `gpu_memory_utilization` set to 0.5 for AR and 0.3 for DiT:

```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
    --model bytedance-research/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
    --enable-diffusion-pipeline-profiler \
    --log-stats \
    --output mammoth_t2i.png
```

#### Verification

The first request took 85.224 seconds. The AR stage generated 4,161 visual tokens in 72.996 seconds, and the DiT stage took 12.163 seconds. AR weight loading used 21.4 GiB and took 8.250 seconds. DiT weight loading used 5.49 GiB and took 1.824 seconds. The largest one second whole device memory sample was 106.57 GiB, including the AR KV cache reserved by the 0.5 memory setting.

The output was a valid 1024 by 1024 RGB PNG.

### Startup benchmark

The [startup benchmark](../../benchmarks/mammoth_moda2/README.md) provides
commands, measurement boundaries and a single-A800 eager baseline. On that
16-CPU-quota machine, parallel stage initialization and 16 CPU threads reduced
`Omni()` initialization from about 60 s to 37 s. See the benchmark for the
configuration and limitations; CUDA-graph mode needs separate memory and
quality validation.

## MammothModa2-Dev unified inference

MammothModa2-Dev uses a Qwen3-VL AR backbone, while MammothModa2-Preview uses
Qwen2.5-VL. vLLM-Omni selects the matching implementation from the nested
`llm_config.model_type`; no checkpoint edits or `trust_remote_code` flag are
required.

Text-to-text and image-to-text use the AR-only deploy. Text-to-image loads the
Qwen3 generation experts (`gen_mlp`), extra visual vocabulary and image head,
then sends the generated visual tokens and hidden states to the DiT stage.

Download the checkpoint:

```bash
hf download bytedance-research/MammothModa2-Dev --local-dir ./MammothModa2-Dev
```

Run text-to-text through the shared understanding example. It recognizes the
Dev checkpoint as MammothModa2 and automatically selects
`mammoth_moda2_ar.yaml`:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Dev \
  --prompt "Explain multimodal generation in three sentences."
```

Add an image for image-to-text or image summarization:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Dev \
  --image ./image.png \
  --prompt "Please summarize the content of this image."
```

The Dev checkpoint is approximately 47.55 GiB on disk. In the verified AR-only
run, loaded model weights used approximately 16.97 GiB of GPU memory before KV
and encoder caches. Allow additional GPU memory for those caches and the input
image.
