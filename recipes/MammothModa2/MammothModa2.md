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
recipe by default. An experimental two-rank Preview DiT Ulysses configuration
is described below; it does not change the single-rank default.

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
stage 0 and 0.3 for stage 1. The A800 validation plan below also shows a
two-GPU placement with one stage per GPU for attributable timing and memory;
the results are pending.

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

### 2x NVIDIA A800 80GB validation plan

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

### Experimental Preview DiT Ulysses SP=2

This opt-in path splits only the main joint-transformer sequence. Q-Former,
text/noise refiners, sequential CFG, scheduler and VAE remain replicated.
Preview has 21 query heads and 7 KV heads with head dimension 120, so degree
two requires `ulysses_mode: advanced_uaa`; strict even-head partitioning is
not valid. Shared Ulysses temporarily pads the head groups and restores the
original output heads. Sequence padding is removed before image extraction.

The initial scope is Preview text-to-image, request mode, one request and
one output image, with eager execution and no cache acceleration,
quantization or offload. Ring, TP/PP/DP, HSDP, expert/CFG/VAE parallelism,
step execution and Dev text-to-image are not supported with this SP path.
The AR-only Preview/Dev understanding topology is unchanged.

The following offline configuration completed a one-prompt Preview E2E smoke
on two A100-SXM4-80GB GPUs with NVLink NV4. AR shares GPU 0 with DiT rank 0;
DiT rank 1 uses GPU 1. This is not a general capacity or performance guarantee:
SP does not shard model weights or the replicated refiners/VAE. Keep the
default deploy config unchanged and save this opt-in config separately as
`mammoth-sp2.yaml`:

```yaml
async_chunk: false
pipeline: mammoth_moda2
trust_remote_code: true
distributed_executor_backend: mp
dtype: bfloat16
enable_prefix_caching: false
stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 1
    max_model_len: 8192
    gpu_memory_utilization: 0.35
    enforce_eager: true
  - stage_id: 1
    devices: "0,1"
    max_num_seqs: 1
    gpu_memory_utilization: 0.3
    enforce_eager: true
    ulysses_degree: 2
    ulysses_mode: advanced_uaa
    diffusion_attention_backend: TORCH_SDPA
    engine_extras:
      dtype: float32
```

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview --deploy-config ./mammoth-sp2.yaml \
  --ulysses-degree 2 --ulysses-mode advanced_uaa --enforce-eager \
  --prompt "A red ceramic teapot on a wooden table beside a small green plant, soft morning light, detailed product photograph." \
  --height 1024 --width 1024 --seed 42 --num-inference-steps 50 \
  --guidance-scale 4.0 --extra-body '{"text_guidance_scale":4.0,"cfg_range":[0.0,1.0]}' \
  --output ./mammoth-sp2.png
```

Pass the Ulysses flags explicitly: the shared example's command-line defaults
also enter config resolution. For the SP=1 control, change stage 1's devices
to `"0"`, set its `ulysses_degree` to 1, and pass `--ulysses-degree 1`.

The correctness control below uses two CUDA devices, real NCCL collectives,
released 2520/21/7/120 head geometry with reduced depth, and FP32 SDPA math.
A separate tiny native pipeline replay exercises the constructor, registry
hooks, Q-Former, DiT, sequential CFG, request-level seed handling, scheduler and
VAE with synthetic AR conditioning. It tests explicit/default-seed A/B/A
requests; it is not a released-conditioning or full-checkpoint E2E test.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m pytest -q \
  tests/diffusion/distributed/test_mammothmoda2_ulysses.py
```

The full-weight smoke used Preview revision
`ef5a5e41dbf0de1ef6275586b7580f0d4248b4c6`, vLLM 0.28.0, Torch 2.13.0+cu130,
Transformers 5.14.1, Diffusers 0.40.0, Python 3.12.3 and driver 580.159.03.
Both SP=1 and SP=2 completed and saved 1024x1024 RGB PNG images. AR token IDs matched;
the saved images differed by at most one 8-bit channel value (mean absolute
difference 0.005415). This is a single-prompt smoke, not raw-latent parity or
a broad image-quality/performance result.

Understanding regressions, BF16 SP accuracy, peak-memory reduction and paired
speedup measurements remain **NOT_RUN**. For a meaningful comparison, keep
code, weights, attention backend, dtype, prompts, seeds and sampling settings
fixed between SP=1 and SP=2.

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
