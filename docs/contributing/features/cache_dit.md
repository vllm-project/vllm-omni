# Support Cache-DiT

This section describes how to add cache-dit acceleration to a new diffusion pipeline. We use the Qwen-Image pipeline and LongCat-Image pipeline as reference implementations.

---

## Table of Contents

- [Overview](#overview)
- [Standard Models: Automatic Support](#standard-models-automatic-support)
- [Custom Architectures: Writing Custom Implementation](#custom-architectures-writing-custom-implementation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview

### What is Cache-DiT?

Cache-DiT is an acceleration library for Diffusion Transformers (DiT) that caches intermediate computation results across denoising steps. The core insight is that adjacent denoising steps often produce similar intermediate features, so we can skip redundant computations by reusing cached results.

The library supports three main caching strategies:

- **DBCache:** Dynamic block-level caching that selectively computes or caches transformer blocks based on residual differences
- **TaylorSeer:** Calibration-based prediction that estimates block outputs using Taylor expansion
- **SCM (Step Computation Masking):** Dynamic step skipping based on configurable policies

### Architecture

vLLM-omni integrates cache-dit through the `CacheDiTBackend` class, which provides a unified interface for managing cache-dit acceleration on diffusion models.

| Method/Class | Purpose | Behavior |
|--------------|---------|----------|
| [`CacheDiTBackend`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/#vllm_omni.diffusion.cache.CacheBackend) | Unified backend interface | Automatically handles enabler selection and cache refresh |
| [`enable_cache_for_dit()`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/cache_dit_backend/#vllm_omni.diffusion.cache.cache_dit_backend.enable_cache_for_dit) | Apply caching to transformer | Configures DBCache on transformer blocks |

**Key APIs from Cache-DiT:**

[Cache-DiT API Reference](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/)

| API | Description |
|-----|-------------|
| `BlockAdapter` | Core abstraction for applying cache-dit to transformers. Specifies transformer module(s), block list(s), and forward signature pattern(s). |
| `ForwardPattern` | Defines block forward signature patterns: `Pattern_0`, `Pattern_1`, `Pattern_2` |
| `ParamsModifier` | Per-transformer or per-block-list cache configuration customization |
| `DBCacheConfig` | Configuration for DBCache parameters (warmup steps, cached steps, thresholds) |
| `refresh_context()` | Update cache context | Called when `num_inference_steps` changes |

---

## Standard Models: Automatic Support

Most DiT models follow this pattern:
- Single transformer with one `ModuleList` of blocks
- Standard forward signature
- Compatible with cache-dit's automatic detection

**Examples:** Qwen-Image, Z-Image

For standard single-transformer models, **no code changes are needed**. The `CacheDiTBackend` automatically uses `enable_cache_for_dit()`:

```python
from vllm_omni import Omni

# Works automatically for standard models
omni = Omni(
    model="Qwen/Qwen-Image",  # Standard single-transformer model
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
    }
)
```

**What happens automatically:**

```python
def enable_cache_for_dit(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Default enabler for standard single-transformer DiT models."""

    # Build cache configuration
    db_cache_config = DBCacheConfig(
        num_inference_steps=None,  # Will be set during first inference
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )

    # Enable cache-dit on transformer
    cache_dit.enable_cache(
        pipeline.transformer,
        cache_config=db_cache_config,
    )

    # Return refresh function for dynamic num_inference_steps updates
    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True):
        cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)

    return refresh_cache_context
```

---

## Custom Architectures: Writing Custom Implementation

Some models require custom handling:

- **Dual-transformer:** Models with separate high-noise and low-noise transformers (e.g., Wan2.2)
- **Multi-block-list:** Models with multiple block lists in one transformer (e.g., LongCatImage with `transformer_blocks` + `single_transformer_blocks`)
- **Special forward patterns:** Models with non-standard block execution patterns

### Example 1: Dual-Transformer Model (Wan2.2)

Wan2.2 uses two transformers: one for high-noise steps and one for low-noise steps.

**Key difference:** Use `BlockAdapter` to wrap multiple transformers with separate configurations.

```python
# Standard: cache_dit.enable_cache(pipeline.transformer, ...)
# Custom: Use BlockAdapter to handle multiple transformers
cache_dit.enable_cache(
    BlockAdapter(
        transformer=[pipeline.transformer, pipeline.transformer_2],  # Multiple transformers
        blocks=[pipeline.transformer.blocks, pipeline.transformer_2.blocks],
        forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
        params_modifiers=[
            ParamsModifier(...),  # Config for high-noise transformer
            ParamsModifier(...),  # Config for low-noise transformer (different params)
        ],
    ),
    cache_config=db_cache_config,
)
```

**Key difference:** `refresh_context` must be called on each transformer separately.

```python
# Standard: cache_dit.refresh_context(pipeline.transformer, num_inference_steps=N)
# Custom: Refresh each transformer with its own step count
def refresh_cache_context(pipeline, num_inference_steps, verbose=True):
    high_steps, low_steps = _split_inference_steps(num_inference_steps)
    cache_dit.refresh_context(pipeline.transformer, num_inference_steps=high_steps, ...)
    cache_dit.refresh_context(pipeline.transformer_2, num_inference_steps=low_steps, ...)
```

### Example 2: Multi-Block-List Model (LongCatImage)

LongCatImage has a single transformer with two block lists: `transformer_blocks` and `single_transformer_blocks`.

**Key difference:** Use `BlockAdapter` to specify multiple block lists within one transformer.

```python
# Standard: cache_dit.enable_cache(pipeline.transformer, ...)
#   - Automatically detects single block list
# Custom: Use BlockAdapter to specify multiple block lists
cache_dit.enable_cache(
    BlockAdapter(
        transformer=pipeline.transformer,  # Single transformer
        blocks=[
            pipeline.transformer.transformer_blocks,        # Block list 1
            pipeline.transformer.single_transformer_blocks, # Block list 2
        ],
        forward_pattern=[ForwardPattern.Pattern_1, ForwardPattern.Pattern_1],
        params_modifiers=[modifier],
    ),
    cache_config=db_cache_config,
)
```

> **Note:** For single transformer with multiple block lists, `refresh_context` works the same as standard models.

### Registering Custom Implementations

After writing your custom enabler, register it in `CUSTOM_DIT_ENABLERS` in `vllm_omni/diffusion/cache/cache_dit_backend.py`:

```python
CUSTOM_DIT_ENABLERS = {
    "Wan22Pipeline": enable_cache_for_wan22,
    "LongCatImagePipeline": enable_cache_for_longcat_image,
    "YourCustomPipeline": enable_cache_for_your_model,  # Add here
}
```

---

## Testing

After adding cache-dit support, test with:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Test your custom model
omni = Omni(
    model="your-model-name",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.24,
    }
)

images = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Verify:**

1. Cache is applied (check logs for "Cache-dit enabled successfully on xxx")
2. Performance improvement (should be around 1.5x-2x faster)
3. Image quality (compare with `cache_backend=None`)

---

## Troubleshooting

### Issue: Cache not applied

**Symptoms:** No speedup observed, no cache-related log messages.

**Causes & Solutions:**

- **Enabler not registered:**

**Problem:** Pipeline name not in `CUSTOM_DIT_ENABLERS` registry.

**Solution:** Verify `pipeline.__class__.__name__` matches the registry key and add your enabler to `CUSTOM_DIT_ENABLERS`.

### Issue: Quality degradation

**Symptoms:** Generated images have artifacts or lower quality compared to non-cached inference.

**Causes & Solutions:**

- **Cache parameters too aggressive:**

**Solution:**
```python
cache_config={
    "residual_diff_threshold": 0.12,  # Lower from 0.24 (try 0.12-0.18)
    "max_warmup_steps": 6,            # Increase from 4 (try 6-8)
    "max_continuous_cached_steps": 2, # Reduce if higher
}
```

Check the [user guide for cache_dit](../../user_guide/diffusion/cache_dit_acceleration.md) for more adjustable parameters.

---

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **Standard DiT** | [`cache_dit_backend.py::enable_cache_for_dit`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/cache_dit_backend/#vllm_omni.diffusion.cache.cache_dit_backend.enable_cache_for_dit) | Default enabler | Single transformer, automatic |
| **Wan2.2** | [`cache_dit_backend.py::enable_cache_for_wan22`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/cache_dit_backend/#vllm_omni.diffusion.cache.cache_dit_backend.enable_cache_for_wan22) | Dual-transformer | Separate high/low noise transformers |
| **LongCat** | [`cache_dit_backend.py::enable_cache_for_longcat_image`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/cache_dit_backend/#vllm_omni.diffusion.cache.cache_dit_backend.enable_cache_for_longcat_image) | Multi-block-list | Two block lists in one transformer |
| **BAGEL** | [`cache_dit_backend.py::enable_cache_for_bagel`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/cache_dit_backend/#vllm_omni.diffusion.cache.cache_dit_backend.enable_cache_for_bagel) | Omni model | Complex architecture |

---

## Summary

Adding cache-dit support:

1. ✅ **Check model type** - Standard models work automatically, custom architectures need enablers
2. ✅ **Write enabler** (if needed) - Use `BlockAdapter` for complex architectures
3. ✅ **Register enabler** (if needed) - Add to `CUSTOM_DIT_ENABLERS` dictionary
4. ✅ **Return refresh function** (if needed) - Handle `num_inference_steps` changes
5. ✅ **Test** - Verify with `cache_backend="cache_dit"`

For most models, the default enabler is sufficient. Only write custom enablers for complex architectures!
