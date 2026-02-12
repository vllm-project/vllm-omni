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
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Simplest way: just enable cache-dit with default parameters
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
)

images = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Default Parameters**: When `cache_config` is not provided, cache-dit uses optimized default values. See the [Configuration Reference](#configuration-reference) section for a complete list of all parameters and their default values.

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

## Online Serving (OpenAI-Compatible)

Enable Cache-DiT for online serving by passing `--cache-backend cache_dit` when starting the server:

```bash
# Use Cache-DiT default (recommended) parameters
vllm serve Qwen/Qwen-Image --omni --port 8091 --cache-backend cache_dit
```

To customize Cache-DiT settings for online serving, pass a JSON string via `--cache-config`:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks": 1, "Bn_compute_blocks": 0, "max_warmup_steps": 4, "residual_diff_threshold": 0.12}'
```

## Acceleration Methods

For comprehensive illustration, please view cache-dit [User_Guide](https://cache-dit.readthedocs.io/en/latest/user_guide/OVERVIEWS/)

### 1. DBCache (Dual Block Cache)

DBCache intelligently caches intermediate transformer block outputs when the residual differences between consecutive steps are small, reducing redundant computations without sacrificing quality.

**Key Parameters**:

- `Fn_compute_blocks` (int, default: 1): Number of **first n** transformer blocks used to compute stable feature differences. Higher values provide more accurate caching decisions but increase computation.
- `Bn_compute_blocks` (int, default: 0): Number of **last n** transformer blocks used for additional fusion. These blocks act as an auto-scaler for approximate hidden states.
- `max_warmup_steps` (int, default: 4): Number of initial steps where caching is disabled to ensure the model learns sufficient features before caching begins. Optimized for few-step distilled models.
- `residual_diff_threshold` (float, default: 0.24): Threshold for residual difference. Higher values lead to faster performance but may reduce precision. Default uses a relatively higher threshold for more aggressive caching.
- `max_cached_steps` (int, default: -1): Maximum number of cached steps. Set to -1 for unlimited caching.
- `max_continuous_cached_steps` (int, default: 3): Maximum number of consecutive cached steps. Limits consecutive caching to prevent precision degradation.

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

- Default `Fn_compute_blocks=1` works well for most cases. Increase to 8-12 for larger models or when more accuracy is needed
- Increase `residual_diff_threshold` (e.g., 0.12-0.15) for faster inference with slight quality trade-off, or decrease from default 0.24 for higher quality
- Default `max_warmup_steps=4` is optimized for few-step models. Increase to 6-8 for more steps if needed

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
  - `None`: SCM disabled (default)
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

- SCM is disabled by default (`scm_steps_mask_policy=None`). Enable it by setting a policy value if you need additional acceleration
- Start with `"medium"` policy and adjust based on quality requirements
- Use `"fast"` or `"ultra"` for maximum speed when quality can be slightly compromised
- `"dynamic"` policy generally provides better quality than `"static"`
- SCM mask is automatically regenerated when `num_inference_steps` changes during inference

## Configuration Reference

### DiffusionCacheConfig Parameters

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

## Example: Accelerate Text-to-Image Generation with CacheDiT

See `examples/offline_inference/text_to_image/text_to_image.py` for a complete working example with cache-dit acceleration.

```bash
# Enable cache-dit with hybrid acceleration
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend cache_dit \
    --num-inference-steps 50
```


The script uses cache-dit acceleration with a hybrid configuration combining DBCache, SCM, and TaylorSeer:

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
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
        "scm_steps_mask_policy": "fast",  # Set to None to disable SCM
        "scm_steps_policy": "dynamic",
    },
)
```

You can customize the configuration by modifying the `cache_config` dictionary to use only specific methods (e.g., DBCache only, DBCache + SCM, etc.) based on your quality and speed requirements.

To test another model, you can modify `--model` with the target model identifier like `Tongyi-MAI/Z-Image-Turbo` and update `cache_config` according the model architecture (e.g., number of transformer blocks).


## Additional Resources

- [Cache-DiT User Guide](https://cache-dit.readthedocs.io/en/latest/user_guide/OVERVIEWS/)
- [Cache-DiT Benchmark](https://cache-dit.readthedocs.io/en/latest/benchmark/HYBRID_CACHE/)
- [DBCache Technical Details](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/)
