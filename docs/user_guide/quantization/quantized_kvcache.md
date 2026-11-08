# Diffusion Attention Runtime Quantization

## Overview

In DiT-based image and video generation, attention can take a large share of
denoising time, especially for high-resolution or long-frame workloads.
vLLM-Omni supports online Q/K/V quantization for eligible diffusion attention
paths while keeping model weights in their original dtype.

Runtime attention precision is configured through the historically named
`diffusion_kv_cache_dtype` field on `OmniDiffusionConfig` (CLI:
`--diffusion-kv-cache-dtype`). Despite the `kv_cache` name, this setting is a
**dense diffusion attention compute selector** in the current implementation;
it does not change the dtype or allocation of a persistent KV cache.

Do not use vLLM's `--kv-cache-dtype` for this feature. That upstream option
controls autoregressive language-model KV-cache storage and is intentionally
separate from Omni's diffusion setting.

In vLLM-Omni diffusion pipelines, this is a runtime FA path: Q/K/V tensors are
quantized for the current attention call and then discarded. It does not
quantize model weights, does not write a persistent FP8 KV buffer, and is
separate from [FP8 W8A8](fp8.md), [Int8 W8A8](int8.md), or pre-quantized
checkpoint formats.

The platform implementations use different quantization runtimes:

- CUDA SM90 calls vLLM's upstream dynamic per-tensor `scaled_fp8_quant` for
  Q, K and V separately. The upstream operator computes the scales and
  converts the inputs to FP8. Omni passes those FP8 tensors and their scales
  to FA3, broadcasting each scalar to `(batch_size, num_kv_heads)`.
- Ascend NPU passes the BF16/FP16 Q/K/V and `precision=fp8`, `mxfp8`, or
  `mxfp4` to MindIE-SD `quant_attention`, which performs the vendor-side
  quantization and attention within that runtime call.

Persistent KV storage quantization is a separate cache lifecycle and is not
part of this option.

CUDA FP8 attention returns the input dtype (BF16 or FP16), casting FA3's BF16
output back to FP16 when needed by the model's output projection. The current
CUDA path requires dense inputs without caller masks, packed-sequence
metadata, or piecewise spans. Dynamic quantization adds reductions,
conversions, and kernel launches, so enabling it can increase latency,
especially for short sequences. Faster generation or lower peak memory use
is not guaranteed.

If `diffusion_kv_cache_dtype` is not set, attention runs in the native dtype.

## Hardware Support

| Device | Dense FP8 | Dense MXFP8/MXFP4 | BSA FP8/MXFP4 |
| -------- | ------------ | ------------------- | ---------------- |
| Ascend NPU | ✅ | ✅ | ✅ |
| NVIDIA GPU (SM90 Hopper) | ✅ | ❌ | ❌ |
| AMD ROCm | ❌ | ❌ | ❌ |
| Intel XPU | ❌ | ❌ | ❌ |

Legend: `✅` supported, `❌` unsupported.

Dense FP8 is available through CUDA FA3 on SM90 and through the NPU Flash
Attention backend. MXFP and BSA variants remain NPU-only. Other backends reject
an incompatible explicit configuration.

## Model Type Support

### Diffusion Model

| Model | Scope | Status | Notes |
| --- | --- | --- | --- |
| Wan2.2 | Self-attention on Ascend NPU | NPU interface and T2V smoke tested | Compare quality and latency against a BF16 baseline before production use |
| Wan2.2 | Self-attention on CUDA SM90 | Full-video E2E exercised | Experimental: quality differences and latency overhead observed |
| Other diffusion models | Eligible DiT full-attention FA on Ascend NPU or CUDA SM90 | Not tested | Validate `diffusion_kv_cache_dtype="fp8"` and the skip-step/layer settings for each model |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

Not tested for FP8 FA. Treat any use as experimental unless a model-specific
guide documents support.

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

Not tested. If the diffusion stage uses a supported Flash Attention backend,
`diffusion_kv_cache_dtype` may apply in theory; validate quality and latency for
each stage and model.

## Configuration

Offline diffusion example:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
    --model <your-wan2.2-model> \
    --prompt "A cat sitting on a surfboard at the beach" \
    --height 1280 \
    --width 720 \
    --num-frames 61 \
    --num-inference-steps 4 \
    --ulysses-degree 4 \
    --vae-patch-parallel-size 4 \
    --diffusion-kv-cache-dtype fp8 \
    --diffusion-kv-cache-skip-steps "0,1" \
    --diffusion-kv-cache-skip-layers "0-2"
```

Online serving:

```bash
vllm serve <your-model> --omni --diffusion-kv-cache-dtype fp8
```

Deploy config:

```yaml
stages:
  - stage_id: 0
    diffusion_kv_cache_dtype: "fp8"
    diffusion_kv_cache_skip_steps: "0,1"
    diffusion_kv_cache_skip_layers: "0-2"
```

The `model_stage` and diffusion execution type belong to the registered
`PipelineConfig`; the deploy YAML only carries runtime overrides.

The legacy keyword aliases `kv_cache_dtype`, `kv_cache_skip_steps`, and
`kv_cache_skip_layers` remain accepted when constructing
`OmniDiffusionConfig` directly. They are not deploy YAML fields; prefer the
`diffusion_*` names for new code.

## Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `diffusion_kv_cache_dtype` | str \| None | `None` | Dense diffusion attention compute precision selector (historical `kv_cache` name; it does not select persistent cache storage). CUDA SM90 dense FA supports `fp8`; NPU dense FA supports `fp8`, `mxfp8`, and `mxfp4`; NPU BSA supports `fp8` and `mxfp4` |
| `diffusion_kv_cache_skip_steps` | str \| None | `None` | Denoising step selector to keep in native dtype, for example `"0,1,4-6"` |
| `diffusion_kv_cache_skip_layers` | str \| None | `None` | Transformer layer selector to keep in native dtype, for example `"0-2,10"` |

Selectors use comma-separated integers and inclusive ranges. Listed steps or
layers run floating-point attention through the selected Dense or BSA backend;
all other eligible forwards use `diffusion_kv_cache_dtype`.

## Validation and Notes

1. Compare generated images or videos against a BF16 baseline with the same
   seed, prompt, resolution, frame count, and denoising steps.
2. Use `diffusion_kv_cache_skip_steps` for denoising steps where quality is more
   sensitive.
3. Use `diffusion_kv_cache_skip_layers` for transformer layers that show visible quality
   regressions.
4. Report both latency and quality results when enabling this option for a new
   model. For image or video models, include visual comparison and quantitative
   metrics when available, such as PSNR or SSIM.

## Wan2.2 T2V quantized attention on Ascend

Wan2.2 T2V A14B supports quantized self-attention through MindIE-SD. Model
weights are unchanged, and model-level opt-out keeps cross-attention in its
native dtype.

| Attention path | self-attention `backend` | Supported `diffusion_kv_cache_dtype` |
| --- | --- | --- |
| Dense FA | `FLASH_ATTN` | `fp8`, `mxfp8`, `mxfp4` |
| BSA | `RAINFUSION_ATTN` | `fp8`, `mxfp4` |

Configure the method and fallback selectors once at the diffusion stage. The
per-role attention config only selects the self-attention backend:

```yaml
stages:
  - stage_id: 0
    diffusion_kv_cache_dtype: "mxfp8"
    diffusion_kv_cache_skip_layers: "0,39"
    diffusion_kv_cache_skip_steps: "0,1,38,39"
    diffusion_attention_config:
      per_role:
        self:
          backend: FLASH_ATTN  # Use RAINFUSION_ATTN with fp8 or mxfp4 for BSA.
```

For 40 denoising steps, `diffusion_kv_cache_skip_steps: "0,1,38,39"` and
`diffusion_kv_cache_skip_layers: "0,39"` are recommended starting points. Steps
are zero-based across the complete request and do not reset when Wan switches
transformers; layer indices are zero-based and local to each transformer.
Selectors accept comma-separated indices and inclusive ranges such as
`"0,3-5"`. A selected forward uses floating-point attention while preserving
the Dense or BSA path.

Unsupported precision/input combinations raise an error and must be corrected
in the configuration; operator errors are not retried with another precision.

For BSA, change the backend to `RAINFUSION_ATTN` and set
`diffusion_kv_cache_dtype` to `fp8` or `mxfp4`.
See [RainFusion attention](../diffusion/attention_backends/rainfusion.md) for
sparse-path configuration and behavior.

### Requirement and interface test

Until these APIs are available in a MindIE-SD release, use revision
[`8637b5333b0225381b215390fd09a8732e671cc4`](https://gitcode.com/zqxu/MindIE-SD/commit/8637b5333b0225381b215390fd09a8732e671cc4).
Build the Python package, PyTorch plugin, and custom operators from that same
checkout. With CANN sourced, run the NPU interface tests from the Omni checkout:

```bash
python -m pytest tests/platforms/npu/quant/test_kv_quant_npu.py \
    -k real_npu -vv -s -o addopts=''
```

A skipped NPU test is not a pass. Validate generated video quality against a
floating-point run with the same prompt, seed, resolution, and step count.
