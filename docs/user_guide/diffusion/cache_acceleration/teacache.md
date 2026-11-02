# TeaCache Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

TeaCache accelerates diffusion model inference by caching transformer computations when consecutive timesteps are similar, providing up to **1.5x-2.0x speedup** with minimal quality loss on models with calibrated coefficients and enough steps to skip (few-step distilled models gain less; see the Summary). It dynamically decides whether to reuse cached outputs based on input similarity, making it ideal for production deployments where inference speed matters without sacrificing generation quality.

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

### Qwen-Image 2.1 (experimental)

The 2.1 single-stream model requires its own calibrated coefficients and an
explicit finite positive `rel_l1_thresh`; neither has a model-specific default.
Passing only `cache_backend="tea_cache"` raises an error; older Qwen-Image
coefficients do not establish compatibility. This path remains experimental. A bounded
single-GPU BF16 eager experiment passed force-full parity and held-out numerical
quality checks; human visual acceptance remains pending.

The initial scope is single-GPU, batch-1 request mode, unquantized weights and
no LoRA. Step mode, parallel execution, FP8 prefix storage and offload are
rejected by both the TeaCache and calibration enablers. Dynamic LoRA wrappers
loaded after initialization are rejected before prefill writes prefix KV. Whole-decode
CUDA graphs are disabled automatically; evaluate compilation separately.

Two optional keys refine the skip decision:

- `cfg_rel_l1_thresh` replaces `rel_l1_thresh` for requests with true CFG.
  Both CFG branches see the same decision feature, so they skip on the same
  steps and the guidance scale amplifies the reused residuals' error.
- `skip_timestep_range` is an exclusive `[low, high]` range of the
  transformer timestep in `[0, 1]`. Decode steps outside it always compute and
  restart the accumulated distance. Output changes are largest in the earliest
  and latest steps, where a single reused residual can fail the quality gate.

After fitting and validating `coefficients` for the exact checkpoint and workload:

```python
omni = Omni(
    model=model_snapshot_path,
    max_num_seqs=1,
    dtype="bfloat16",
    enforce_eager=True,
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.10,
        "cfg_rel_l1_thresh": 0.05,
        "skip_timestep_range": [0.30, 0.855],
        "coefficients": coefficients,
    },
)
```

The timestep range is tied to the measured schedule: 50 explicit linear sigmas
at 1024x1024, where the scheduler's dynamic shift maps denoising steps 13–40
(0-based; the prefill is step 0) into `(0.30, 0.855)`. Other resolutions, step counts or sigma schedules shift those
timesteps; recompute the range and revalidate quality for them.

#### Measured configuration and limits

The experiment used source `36500620c3b3cd13c51b32f2fd4b26dd77eeca75`,
`Qwen/Qwen-Image-2.1` revision `d26bb61231c349cf6b7896fa83353113880e1ba3`,
one RTX PRO 6000 Blackwell Server Edition (97887 MiB, SM120), driver 595.58.03,
Torch 2.13.0+cu130, vLLM 0.32.0, Diffusers 0.40.0, Transformers 5.19.0 and
`CUDNN_ATTN`. Requests used BF16, 1024-square RGBA output, 50 explicit sigmas
`[1 - i / 50 for i in range(50)]`, request mode and concurrency/batch size 1.

Twelve training prompts, two training seeds and CFG 1/4 produced 3456 dense
observations. The frozen merged quartic, highest degree first, was:

```python
coefficients = [
    489544.41506285727,
    -65325.67451621187,
    3012.825475742039,
    -55.92014411118746,
    0.4159928016209747,
]
```

These are experiment coefficients, not registered defaults. Eight separate
held-out prompts, two held-out seeds and CFG 1/4 formed 32 comparisons per
threshold. Public reference images were reused across splits; this does not
test generalization to unseen reference images. Only threshold 0.05 passed
every comparison in both CFG cohorts. Thresholds 0.10, 0.15 and 0.20 failed
some cases and are not qualified by this experiment.

Each RGB, alpha and black/white-composited view required PSNR at least 30 dB
and SSIM at least 0.95; RGB and composited views also required LPIPS at most
0.10. Alpha had no LPIPS gate. Worst results across each CFG cohort were:

| CFG | Passing cases | Minimum PSNR dB | Minimum SSIM | Maximum LPIPS |
| --- | --- | --- | --- | --- |
| 1 | 16/16 | 38.158 | 0.99487 | 0.00266 |
| 4 | 16/16 | 30.632 | 0.98604 | 0.00535 |

Native HTTP measurements covered four layouts (text, RGBA output, one-reference
editing and four-reference input), CFG 1/4 and seed 2026. Each layout/configuration
discarded one full 50-step warmup and retained two requests. TeaCache 0.05 reduced
mean end-to-end latency by 7.2–11.7% relative to the matched eager no-cache
baseline in these eight layouts. Two samples per layout do not establish tail
latency, concurrency throughput or results on other GPUs/workloads.

The default compile/decode-graph baseline failed the declared numerical gate
in three CFG4 layouts on this stack. Those rows are not equal-quality speed
comparisons. TeaCache was slower than that default in the two text/RGBA CFG4
layouts, so there is no universal speedup over the project default. A separate
runtime-only numerical diagnostic does not qualify compiled TeaCache or change
the configuration measured here. Check typography, editing identity, reference
placement and transparent edges before accepting the numerical result.

The measurements above use no timestep window. The skip policy keys were
checked afterwards on the same GPU, checkpoint, held-out comparisons, gates and
frozen coefficients, with `skip_timestep_range=[0.30, 0.855]`. Latency is the
summed per-request in-process generation time against the matched dense eager
run; each comparison ran once.

| Configuration | CFG | Passing | Decode steps skipped | Minimum PSNR dB | Latency |
| --- | --- | --- | --- | --- | --- |
| `rel_l1_thresh=0.10` | 1 | 16/16 | 32.5% | 31.13 | −30.3% |
| `cfg_rel_l1_thresh=0.05` | 4 | 16/16 | 7.8% | 30.63 | −5.3% |
| `cfg_rel_l1_thresh=0.06` | 4 | 14/16 | 14.0% | 29.06 | −12.6% |

At CFG4 with 0.06, both failures are one structure prompt whose PSNR is
29.1–29.3 dB while SSIM stays at least 0.978 and LPIPS at most 0.008. Only 0.05
passed every true-CFG comparison in this workload. Skips occurred only at
denoising steps 13–40 (0-based; prefill is step 0); every step outside the
window computed. CFG1 outputs were bitwise identical to the earlier exploration
that forced steps at or below 12 and at or above 41 to compute.

These single-run timings are not HTTP, tail-latency or throughput results.
Repeat timing with a matched dense eager baseline and review all 48 confirmation
images before drawing broader quality or performance conclusions.

For an already-loaded pipeline in a single-GPU development harness, install the
dense collector before running calibration requests:

```python
from vllm_omni.diffusion.models.qwen_image_21.teacache import (
    enable_qwen_image21_calibration,
    fit_qwen_image21_coefficients,
)

collector = enable_qwen_image21_calibration(pipeline, max_observations=10000)
# Run complete requests with the normal vLLM/Omni forward and distributed
# contexts, under torch.inference_mode(). The collector never skips blocks.
# Do not use training/held-out prompts or seeds interchangeably.
coefficients = fit_qwen_image21_coefficients(collector.observations)
```

The installation snippet requires a loaded development pipeline; it is not a
standalone model loader or an Omni worker RPC. The collector retains only the
previous feature/output per branch and bounded scalar distances. Prefill resets
tensor history, so no pair spans requests; scalar records remain until the
collector is discarded. Fitting rejects insufficient, non-finite or degenerate
data. A finite fit still needs held-out image quality validation, including CFG,
text, editing and alpha. Record the input-distance range and do not assume the
quartic is reliable outside it.

First compare no-cache inference against a force-full diagnostic configuration:
`{"rel_l1_thresh": 0.2, "coefficients": [0, 0, 0, 0, 1]}`. This constant forces
every decode to compute and is **not a calibrated skip polynomial**. Stop on any
parity failure before measuring approximate skips. Keep checkpoint, input,
seed, resolved sigma schedule, steps, CFG, precision and attention fixed.

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

## Configuration Parameters

In `OmniDiffusionConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rel_l1_thresh` | float | `0.2` | Similarity threshold for cache reuse. Lower values prioritize quality (less caching), higher values prioritize speed (more caching). Suggested range: 0.1-0.3; higher values mostly cost quality on calibrated models |
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
- Scenarios where a model-dependent 1.2-2x speedup is valuable
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
1. Increase the threshold to enable more aggressive caching (non-distilled models only; on few-step distilled models such as Z-Image-Turbo a higher threshold mostly costs quality, see the Summary):
   ```python
   cache_config={"rel_l1_thresh": 0.3}
   ```
2. Ensure you're using sufficient inference steps (35+ recommended for non-distilled models; few-step distilled models cannot add steps)
3. Check that your model architecture is supported (see Supported Models section)

---


## Summary

1. ✅ **Enable TeaCache** - Set `cache_backend="tea_cache"` to get up to 1.5x-2.0x speedup with optimized defaults on 50-step runs; few-step distilled models (e.g. Z-Image-Turbo at 25 steps) change too much per step to skip many, so expect ~1.1x there
2. ✅ **(Optional) Customize** - Adjust thresholds and polynomial coefficients for specific speed/quality trade-offs
