# TeaCache Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Wan VACE](#wan-vace)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

TeaCache accelerates diffusion model inference by caching transformer computations when consecutive timesteps are similar, providing **1.5x-2.0x speedup** with minimal quality loss. It dynamically decides whether to reuse cached outputs based on input similarity, making it ideal for production deployments where inference speed matters without sacrificing generation quality.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

---

## Quick Start



### Basic Usage


```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### Custom Configuration

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2,  # Controls speed/quality tradeoff
    },
)
```

### Using Environment Variable

You can also enable TeaCache via environment variable:

```bash
export DIFFUSION_CACHE_BACKEND=tea_cache
```

Then initialize without explicitly setting `cache_backend`:

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_config={"rel_l1_thresh": 0.2}
)
```

---

## Example Script

### Offline Inference

Use python script under `examples/offline_inference/text_to_image/` or `examples/offline_inference/image_to_image/` with CLI:

```bash
# Text-to-image example
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --cache-backend tea_cache

# Image-to-image example
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-Edit \
  --image input.png \
  --prompt "Edit description" \
  --cache-backend tea_cache \
  --tea-cache-rel-l1-thresh 0.25
```

See the [text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_image/text_to_image.py) or [image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py) for detailed configuration options.

### Online Serving

```bash
# Default configuration
vllm serve Qwen/Qwen-Image --omni --port 8091 --cache-backend tea_cache

# Custom configuration
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.2}'
```

---

## Wan VACE

Wan2.2 VACE supports TeaCache for both one-loaded-expert configurations and
the high-noise/low-noise dual-transformer layout. Each loaded transformer owns
an independent cache state, so accumulated distances and residuals do not
cross the noise boundary.

VACE conditioning blocks are recomputed on every denoising step. TeaCache uses
a compact signature of the current VACE hints as an additional cache gate and
reuses the main-transformer residual only when both the timestep modulation and
VACE conditioning remain similar. This avoids making cache decisions from the
main block input alone, which does not contain hints injected later in the
transformer.

Under sequence parallelism, the cache gate takes the maximum distance across
the SP group so every rank makes the same skip/recompute decision. Only the
scalar decision signal is reduced; VACE hint activations remain sharded.

```bash
vllm serve Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers --omni --port 8090 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.2}'
```

The default Wan VACE polynomial currently uses the Wan2.1-T2V-14B
coefficients as a preliminary starting point. For a calibrated deployment,
pass five workload-specific coefficients through `cache_config`. Start with
`rel_l1_thresh=0.1` when source-video adherence is more important than speed,
then compare against no-cache output before increasing it.

### Benchmarking

Use `benchmarks/diffusion/wan_vace_teacache.py` to compare configurations with
the issue workload (1280×736, 61 frames, 20 steps, guidance 5.0, boundary
ratio 0.875, flow shift 3.0, and seed 1). The request benchmark enforces one
warmup plus at least three measured requests and records client/server latency,
per-stage durations, peak accelerator memory, response hashes, exact commands,
and generated clips.

For a server that can load the model locally, the `matrix` subcommand runs the
complete comparison sequentially. It refuses to reuse an already-running port,
stores one server log per configuration, and terminates each server before
starting the next one. It also raises the synchronous video endpoint timeout
above the client timeout because the server default is only 600 seconds:

```bash
python benchmarks/diffusion/wan_vace_teacache.py matrix \
  --model Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers \
  --input-video /path/to/source.mp4 \
  --server-hardware "<count>x <accelerator> <VRAM>" \
  --server-software "torch=<version>" \
  --server-software "vllm-omni=<commit>" \
  --server-args='--tensor-parallel-size 1' \
  --dinov2-model facebook/dinov2-base
```

This runs no cache, TeaCache thresholds 0.1 and 0.2, and Cache-DiT with the
same request parameters. Add `--dry-run` to print the exact server and request
commands without starting any process, or `--skip-quality` when OpenCV/DINOv2
quality analysis will be run separately.

Start one server configuration at a time. For example:

```bash
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7260 \
  vllm serve Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers --omni --port 8090 \
  --enable-diffusion-pipeline-profiler \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh":0.2}'
```

From another terminal, run the fixed request matrix:

```bash
python benchmarks/diffusion/wan_vace_teacache.py request \
  --base-url http://localhost:8090 \
  --input-video /path/to/source.mp4 \
  --label tea_0_2 \
  --model Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers \
  --server-command "vllm serve Pyros13/Wan2.2-VACE-Fun-A14B-Diffusers --omni --port 8090 --enable-diffusion-pipeline-profiler --cache-backend tea_cache --cache-config '{\"rel_l1_thresh\":0.2}'" \
  --server-hardware "<count>x Ascend 910B3 64GB" \
  --server-software "torch=<version>" \
  --server-software "vllm-omni=<commit>"
```

Restart and repeat for no cache, TeaCache thresholds 0.1 and 0.2, and
Cache-DiT. Keep every other server and request setting fixed. Then compare the
measured artifacts, including no-cache self-run variance:

```bash
python benchmarks/diffusion/wan_vace_teacache.py quality \
  --source-video /path/to/source.mp4 \
  --baseline-video wan_vace_teacache_results/no_cache/measured_01.mp4 \
  --baseline-repeat no_cache_02=wan_vace_teacache_results/no_cache/measured_02.mp4 \
  --baseline-repeat no_cache_03=wan_vace_teacache_results/no_cache/measured_03.mp4 \
  --candidate tea_0_1=wan_vace_teacache_results/tea_0_1/measured_01.mp4 \
  --candidate tea_0_2=wan_vace_teacache_results/tea_0_2/measured_01.mp4 \
  --candidate cache_dit=wan_vace_teacache_results/cache_dit/measured_01.mp4 \
  --dinov2-model facebook/dinov2-base
```

The report's `frame_pearson` is a transparent pixel statistic, not an assumed
reproduction of the unspecified `corr` metric in issue #5079. Use per-frame
DINOv2 cosine, the same-seed no-cache comparison, source adherence, and visual
review together when deciding whether a threshold is acceptable.

---

## Configuration Parameters

In `OmniDiffusionConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rel_l1_thresh` | float | `0.2` | Similarity threshold for cache reuse. Lower values prioritize quality (less caching), higher values prioritize speed (more caching). Suggested range: 0.1-0.8 |
| `coefficients` | list[float] \| None | `None` | Polynomial coefficients for rescaling L1 distance. Must contain exactly 5 elements if provided. If `None`, uses model-specific defaults based on transformer type. |

Users can find the default model coefficients in [`vllm_omni/diffusion/cache/teacache/config.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/cache/teacache/config.py), for example:

```python
_MODEL_COEFFICIENTS = {
    # Qwen-Image transformer coefficients from ComfyUI-TeaCache
    # Tuned specifically for Qwen's dual-stream transformer architecture
    # Used for all Qwen-Image Family pipelines, in general
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    ...
}
```

---

## Best Practices

### When to Use

**Good for:**

- Production deployments requiring faster inference, tolerant of minimal quality loss
- Scenarios where 1.5-2x speedup is valuable
- Useful for single-card acceleration

**Not for:**

- Maximum quality requirements where no degradation is acceptable
- Very short inference runs (< 20 steps) where caching overhead may outweigh benefits


---

## Troubleshooting

### Common Issue 1: Quality Degradation

**Symptoms**: Generated images show artifacts, reduced detail, or inconsistent quality compared to non-cached results

**Solution**:

```python
# Lower the threshold for more conservative caching
cache_config={"rel_l1_thresh": 0.1}
```

### Common Issue 2: Limited Speedup

**Symptoms**: Actual speedup is less than expected (< 1.3x)

**Solutions**:
1. Increase the threshold to enable more aggressive caching:
   ```python
   cache_config={"rel_l1_thresh": 0.8}
   ```
2. Ensure you're using sufficient inference steps (35+ recommended)
3. Check that your model architecture is supported (see Supported Models section)

---


## Summary

1. ✅ **Enable TeaCache** - Set `cache_backend="tea_cache"` to get 1.5x-2.0x speedup with optimized defaults
2. ✅ **(Optional) Customize** - Adjust thresholds and polynomial coefficients for specific speed/quality trade-offs
