# Cache-DiT Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Acceleration Methods](#acceleration-methods)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)
- [Additional Resources](#additional-resources)

---

## Overview

Cache-DiT accelerates diffusion transformer models through intelligent caching mechanisms, providing significant speedup with minimal quality loss. It supports multiple acceleration techniques that can be combined for optimal performance:

- **DBCache**: Dual Block Cache for reducing redundant computations
- **TaylorSeer**: Taylor expansion-based forecasting for faster inference
- **SCM**: Step Computation Masking for selective step computation

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

---

## Quick Start

### Basic Usage

Enable cache-dit acceleration by simply setting `cache_backend="cache_dit"`:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",  # Enable Cache-DiT with defaults
)

outputs = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Note**: When `cache_config` is not provided, Cache-DiT uses optimized default values. See the [Configuration Parameters](#configuration-parameters) section for details.

### Custom Configuration

To customize cache-dit settings, provide a `cache_config` dictionary, for example:

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.12,
    },
)
```

---

## Example Script

### Offline Inference

Use the example script under `examples/offline_inference/text_to_image`:

```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend cache_dit \
    --num-inference-steps 50
```

See the [text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_image/text_to_image.py) for detailed configuration options.

The script uses cache-dit acceleration with a hybrid configuration combining DBCache, SCM, and TaylorSeer:

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        # Scheme: Hybrid DBCache + SCM + TaylorSeer
        "Fn_compute_blocks": 1,  # Optimized for single-transformer models
        "Bn_compute_blocks": 0,  # Number of backward compute blocks
        "max_warmup_steps": 4,  # Maximum warmup steps (works for few-step models)
        "residual_diff_threshold": 0.24,  # Higher threshold for more aggressive caching
        "max_continuous_cached_steps": 3,  # Limit to prevent precision degradation
        # TaylorSeer parameters [cache-dit only]
        "enable_taylorseer": False,  # Disabled by default (not suitable for few-step models)
        "taylorseer_order": 1,  # TaylorSeer polynomial order
        # SCM (Step Computation Masking) parameters [cache-dit only]
        "scm_steps_mask_policy": None,  # SCM mask policy: None (disabled), "slow", "medium", "fast", "ultra"
        "scm_steps_policy": "dynamic",  # SCM steps policy: "dynamic" or "static"
    }
)
```

You can customize the configuration by modifying the `cache_config` dictionary to use only specific methods (e.g., DBCache only, DBCache + SCM, etc.) based on your quality and speed requirements.

For image-to-image tasks, use the example script under `examples/offline_inference/image_to_image`:

```bash
cd examples/offline_inference/image_to_image
python image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --prompt "make the sky more colorful" \
    --image path/to/input/image.jpg \
    --cache-backend cache_dit \
    --num-inference-steps 50 \
    --cache-dit-max-continuous-cached-steps 3 \
    --cache-dit-residual-diff-threshold 0.24 \
    --cache-dit-enable-taylorseer
```

See the [image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py) for detailed configuration options.

### Online Serving

```bash
# Default configuration (recommended)
vllm serve Qwen/Qwen-Image --omni --port 8091 --cache-backend cache_dit

# Custom configuration
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks": 1, "residual_diff_threshold": 0.12}'
```

---

## Acceleration Methods

For comprehensive illustration, please view Cache-DiT [User Guide](https://cache-dit.readthedocs.io/en/latest/user_guide/OVERVIEWS/).

### 1. DBCache (Dual Block Cache)

DBCache intelligently caches intermediate transformer block outputs when the residual differences between consecutive steps are small, reducing redundant computations without sacrificing quality.

**Example Configuration**:

```python
cache_config={
    "Fn_compute_blocks": 8,           # Use first 8 blocks for difference computation
    "Bn_compute_blocks": 0,           # No additional fusion blocks
    "max_warmup_steps": 8,            # Cache after 8 warmup steps
    "residual_diff_threshold": 0.12,  # Lower threshold for faster inference
    "max_cached_steps": -1,           # No limit on cached steps
}
```

**Performance Tips**:

- Default `Fn_compute_blocks=1` works well for most cases. Some models (e.g., FLUX.2-klein) use a larger value for `Fn_compute_blocks` for a balanced performance.
- Increase `residual_diff_threshold` (e.g., 0.12-0.15) for faster inference with slight quality trade-off, or decrease from default 0.24 for higher quality.
- Default `max_warmup_steps=4` is optimized for few-step models. Increase to 6-8 for more steps if needed.

### 2. TaylorSeer

TaylorSeer uses Taylor expansion to forecast future hidden states, allowing the model to skip some computation steps while maintaining quality.

**Example Configuration**:

```python
cache_config={
    "enable_taylorseer": True,
    "taylorseer_order": 1,  # First-order Taylor expansion
}
```

**Performance Tips**:

- TaylorSeer is **not suitable for few-step distilled models**.
- Use `taylorseer_order=1` for most cases (good balance of speed and quality).
- Combine with DBCache for maximum acceleration.
- Higher orders (2-3) may improve quality but reduce speed gains.

### 3. SCM (Step Computation Masking)

SCM allows you to specify which steps must be computed and which can use cached results, similar to LeMiCa/EasyCache style acceleration.

`scm_steps_mask_policy` options (number of compute steps out of 28):

| Policy | Compute Steps | Speed | Quality |
|--------|--------------|-------|---------|
| `None` (default) | All | Baseline | Best |
| `"slow"` | 18 / 28 | Moderate | High |
| `"medium"` | 15 / 28 | Balanced | Good |
| `"fast"` | 11 / 28 | Fast | Moderate |
| `"ultra"` | 8 / 28 | Fastest | Lower |

**Example Configuration**:

```python
cache_config={
    "scm_steps_mask_policy": "medium",  # Balanced speed/quality
    "scm_steps_policy": "dynamic",      # Use dynamic cache
}
```

**Performance Tips**:

- SCM is disabled by default. Enable it by setting a policy value if you need additional acceleration.
- Start with `"medium"` policy and adjust based on quality requirements.
- Use `"fast"` or `"ultra"` for maximum speed when quality can be slightly compromised.
- `"dynamic"` policy generally provides better quality than `"static"`.
- SCM mask is automatically regenerated when `num_inference_steps` changes during inference.

---

## Configuration Parameters

In `cache_config` passed to `Omni` constructor, it accepts the arguments of `DBCacheConfig` ([Cache-DiT API Reference](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/)). Key parameters are listed below:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Fn_compute_blocks` | int | 1 | First n blocks for difference computation (optimized for single-transformer models) |
| `Bn_compute_blocks` | int | 0 | Last n blocks for fusion |
| `max_warmup_steps` | int | 4 | Steps before caching starts (optimized for few-step distilled models) |
| `max_cached_steps` | int | -1 | Max cached steps (-1 = unlimited) |
| `max_continuous_cached_steps` | int | 3 | Max consecutive cached steps (prevents precision degradation) |
| `residual_diff_threshold` | float | 0.24 | Residual difference threshold (higher for more aggressive caching) |
| `num_inference_steps` | int \| None | None | Initial inference steps for SCM mask generation (optional, auto-refreshed during inference) |
| `enable_taylorseer` | bool | False | Enable TaylorSeer acceleration (not suitable for few-step distilled models) |
| `taylorseer_order` | int | 1 | Taylor expansion order |
| `scm_steps_mask_policy` | str \| None | None | SCM mask policy (None, "slow", "medium", "fast", "ultra") |
| `scm_steps_policy` | str | "dynamic" | SCM computation policy ("dynamic" or "static") |

---

## Best Practices

### When to Use

**Good for:**

- Production deployments requiring fast inference
- Diffusion transformer models (DiT architecture)
- Scenarios where 1.5x-3x speedup is valuable

**Not for:**

- Non-DiT architectures (use model-specific acceleration instead)
- Models already using few-step distillation (< 10 steps)

---

## Troubleshooting

### Common Issue 1: Quality Degradation

**Symptoms**: Generated images have visible artifacts or lower quality

**Solution**:
```python
# Reduce aggressiveness - use more conservative settings
cache_config={
    "residual_diff_threshold": 0.20,  # Lower threshold (closer to default 0.24)
    "Fn_compute_blocks": 8,            # Use more blocks for better decisions
    "max_warmup_steps": 6,             # Longer warmup
    "scm_steps_mask_policy": "slow",   # More compute steps
}
```

---

## Summary

Using Cache-DiT acceleration:

1. ✅ **Enable Cache-DiT** - Set `cache_backend="cache_dit"` to get 1.5x-3x speedup with optimized defaults
2. ✅ **(Optional) Customize** - Adjust `cache_config` parameters for specific speed/quality trade-offs
