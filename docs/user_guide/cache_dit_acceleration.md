# Cache-DiT Acceleration Guide

This guide explains how to use cache-dit acceleration in vLLM-Omni to speed up diffusion model inference.

## Overview

Cache-dit is a library that accelerates diffusion transformer models through intelligent caching mechanisms. It supports multiple acceleration techniques that can be combined for optimal performance:

- **DBCache**: Dual Block Cache for reducing redundant computations
- **TaylorSeer**: Taylor expansion-based forecasting for faster inference
- **SCM**: Step Computation Masking for selective step computation

## Quick Start

### Basic Usage

Enable cache-dit acceleration by simply setting `cache_backend="cache_dit"`. Cache-dit will use its recommended default parameters:

```python
from vllm_omni.entrypoints.omni import Omni

# Simplest way: just enable cache-dit with default parameters
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
)

images = omni.generate(
    "a beautiful landscape",
    num_inference_steps=50,
)
```

**Default Parameters**: When `cache_config` is not provided, cache-dit uses:
- `Fn_compute_blocks=8` (aligned with cache-dit's recommended default)
- `Bn_compute_blocks=0`
- `max_warmup_steps=8`
- `residual_diff_threshold=0.08`
- No TaylorSeer or SCM enabled

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

### Example Script

See `examples/offline_inference/qwen_image/text_to_image.py` for a complete working example with cache-dit acceleration.

```bash
# Enable cache-dit with default parameters
cd examples/offline_inference/qwen_image
python text_to_image.py \
    --prompt "a cup of coffee on the table" \
    --enable_cache_dit \
    --num_inference_steps 50
```

The `--enable_cache_dit` flag enables cache-dit acceleration with these customized parameters:

```python
omni = Omni(
    ...
    cache_backend="cache_dit" if args.enable_cache_dit else None,
    cache_config={
        # Scheme: Hybrid DBCache + SCM + TaylorSeer
        # DBCache
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.12,
        # TaylorSeer
        "enable_taylorseer": True,
        "taylorseer_order": 1,
        # SCM
        "scm_steps_mask_policy": "fast",
        "scm_steps_policy": "dynamic",
    },
)

```

## Acceleration Methods

For comprehensive illustration, please view cache-dit [User_Guide](https://github.com/vipshop/cache-dit/blob/main/docs/User_Guide.md)

### 1. DBCache (Dual Block Cache)

DBCache intelligently caches intermediate transformer block outputs when the residual differences between consecutive steps are small, reducing redundant computations without sacrificing quality.

**Key Parameters**:

- `Fn_compute_blocks` (int, default: 1): Number of **first n** transformer blocks used to compute stable feature differences. Higher values provide more accurate caching decisions but increase computation.
- `Bn_compute_blocks` (int, default: 0): Number of **last n** transformer blocks used for additional fusion. These blocks act as an auto-scaler for approximate hidden states.
- `max_warmup_steps` (int, default: 8): Number of initial steps where caching is disabled to ensure the model learns sufficient features before caching begins.
- `residual_diff_threshold` (float, default: 0.08): Threshold for residual difference. Higher values lead to faster performance but may reduce precision.
- `max_cached_steps` (int, default: -1): Maximum number of cached steps. Set to -1 for unlimited caching.
- `max_continuous_cached_steps` (int, default: -1): Maximum number of consecutive cached steps. Set to -1 for unlimited consecutive caching.

**Example Configuration**:

```python
cache_config={
    "Fn_compute_blocks": 8,      # Use first 8 blocks for difference computation
    "Bn_compute_blocks": 0,       # No additional fusion blocks
    "max_warmup_steps": 8,        # Cache after 8 warmup steps
    "residual_diff_threshold": 0.12,  # Higher threshold for faster inference
    "max_cached_steps": -1,        # No limit on cached steps
}
```

**Performance Tips**:
- Start with `Fn_compute_blocks=8` and adjust based on your model size
- Increase `residual_diff_threshold` (e.g., 0.12-0.15) for faster inference with slight quality trade-off
- Reduce `max_warmup_steps` (e.g., 4-6) for faster startup, but ensure sufficient warmup

### 2. TaylorSeer

TaylorSeer uses Taylor expansion to forecast future hidden states, allowing the model to skip some computation steps while maintaining quality.

**Key Parameters**:

- `enable_taylorseer` (bool, default: False): Enable TaylorSeer acceleration
- `taylorseer_order` (int, default: 1): Order of Taylor expansion. Higher orders provide better accuracy but require more computation.

**Example Configuration**:

```python
cache_config={
    "enable_taylorseer": True,
    "taylorseer_order": 1,  # First-order Taylor expansion
}
```

**Performance Tips**:
- Use `taylorseer_order=1` for most cases (good balance of speed and quality)
- Combine with DBCache for maximum acceleration
- Higher orders (2-3) may improve quality but reduce speed gains

### 3. SCM (Step Computation Masking)

SCM allows you to specify which steps must be computed and which can use cached results, similar to LeMiCa/EasyCache style acceleration.

**Key Parameters**:

- `scm_steps_mask_policy` (str | None, default: None): Predefined mask policy. Options:
  - `"slow"`: More compute steps, higher quality (18 compute steps out of 28)
  - `"medium"`: Balanced (15 compute steps out of 28)
  - `"fast"`: More cache steps, faster inference (11 compute steps out of 28)
  - `"ultra"`: Maximum speed (8 compute steps out of 28)
- `scm_steps_policy` (str, default: "dynamic"): Policy for cached steps:
  - `"dynamic"`: Use dynamic cache for masked steps (recommended)
  - `"static"`: Use static cache for masked steps

**Example Configuration**:

```python
cache_config={
    "scm_steps_mask_policy": "medium",  # Balanced speed/quality
    "scm_steps_policy": "dynamic",      # Use dynamic cache
}
```

**Performance Tips**:
- Start with `"medium"` policy and adjust based on quality requirements
- Use `"fast"` or `"ultra"` for maximum speed when quality can be slightly compromised
- `"dynamic"` policy generally provides better quality than `"static"`
- SCM mask is automatically regenerated when `num_inference_steps` changes during inference

## Complete Configuration Examples

### Example 1: DBCache Only (Recommended Starting Point)

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 8,
        "residual_diff_threshold": 0.08,
    },
)
```

### Example 2: DBCache + TaylorSeer (Balanced Speed/Quality)

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        # DBCache
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 8,
        "residual_diff_threshold": 0.12,
        # TaylorSeer
        "enable_taylorseer": True,
        "taylorseer_order": 1,
    },
)
```

### Example 3: DBCache + SCM

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        # DBCache
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.12,
        # SCM
        "scm_steps_mask_policy": "fast",
        "scm_steps_policy": "dynamic",
    },
)
```

### Example 4: All Methods Combined

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        # DBCache
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.12,
        # TaylorSeer
        "enable_taylorseer": True,
        "taylorseer_order": 1,
        # SCM
        "scm_steps_mask_policy": "fast",
        "scm_steps_policy": "dynamic",
    },
)
```

## Dynamic num_inference_steps

Cache-dit automatically handles changes in `num_inference_steps` during inference. When you call `generate()` with a different `num_inference_steps` than the previous request, cache-dit will:

1. Automatically refresh the cache context
2. Regenerate SCM masks if SCM is enabled
3. Update internal configurations to match the new step count

**Example**:

```python
# First request with 50 steps
images1 = omni.generate("prompt 1", num_inference_steps=50)

# Second request with 28 steps - cache context automatically refreshes
images2 = omni.generate("prompt 2", num_inference_steps=28)
```

No manual intervention is required - the refresh happens automatically and efficiently.

## Configuration Reference

### DiffusionCacheConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Fn_compute_blocks` | int | 8 | First n blocks for difference computation (aligned with cache-dit's default) |
| `Bn_compute_blocks` | int | 0 | Last n blocks for fusion |
| `max_warmup_steps` | int | 8 | Steps before caching starts |
| `max_cached_steps` | int | -1 | Max cached steps (-1 = unlimited) |
| `max_continuous_cached_steps` | int | -1 | Max consecutive cached steps (-1 = unlimited) |
| `residual_diff_threshold` | float | 0.08 | Residual difference threshold |
| `num_inference_steps` | int \| None | None | Initial inference steps for SCM mask generation (optional, auto-refreshed during inference) |
| `enable_taylorseer` | bool | False | Enable TaylorSeer acceleration |
| `taylorseer_order` | int | 1 | Taylor expansion order |
| `scm_steps_mask_policy` | str \| None | None | SCM mask policy ("slow", "medium", "fast", "ultra") |
| `scm_steps_policy` | str | "dynamic" | SCM computation policy ("dynamic" or "static") |

## Tips and Best Practices

### 1. Start Simple
The easiest way to get started is to enable cache-dit without any configuration:

```python
omni = Omni(model="Qwen/Qwen-Image", cache_backend="cache_dit")
```

This uses cache-dit's recommended defaults (`Fn_compute_blocks=8`). Once you understand the baseline performance, you can customize the configuration to add other acceleration methods or fine-tune parameters.

### 2. Quality vs Speed Trade-offs
- **Higher quality**: Lower `residual_diff_threshold` (0.08), more `Fn_compute_blocks` (8-12), SCM `"slow"` or `"medium"`
- **Faster inference**: Higher `residual_diff_threshold` (0.12-0.15), fewer `max_warmup_steps` (4-6), SCM `"fast"` or `"ultra"`

### 3. Model-Specific Tuning
Different models may benefit from different configurations:
- **Smaller models**: May work well with `Fn_compute_blocks=4-6`
- **Larger models**: Often benefit from `Fn_compute_blocks=8-12`
- **High-resolution images**: May need lower `residual_diff_threshold` for quality

### 4. Combining Methods
- **DBCache + TaylorSeer**: Good balance, works well for most cases
- **DBCache + SCM**: Can try when quality can be slightly compromised
- **All three**: Maximum acceleration if the parameters are carefully tuned

## Additional Resources

- [Cache-DiT User Guide](https://github.com/vipshop/cache-dit/blob/main/docs/User_Guide.md)
- [Cache-DiT Benchmark](https://github.com/vipshop/cache-dit/tree/main/bench)
- [DBCache Technical Details](https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md)
