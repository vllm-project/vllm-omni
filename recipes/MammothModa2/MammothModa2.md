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
batching, step execution, continuous batching, compilation, quantization,
parallelism, and offload are not enabled by this recipe. TeaCache acceleration
is supported for the DiT stage.

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
against the declared `extra_body_params`.

TeaCache can be enabled for the DiT stage with the same user-facing sampling
parameters:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --guidance-scale 4.0 \
  --num-inference-steps 50 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.1}' \
  --extra-body '{"cfg_range": [0.0, 1.0]}' \
  --output mammoth_t2i_teacache.png
```

The bundled TeaCache coefficients were fitted from MammothModa2 full-compute
traces. For the evaluated 1024x1024, 50-step configuration,
`rel_l1_thresh=0.1` provided the selected quality/speed tradeoff.

The model-specific keys are declared in
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
