# Cache-DiT Patterns Reference

## Overview

Cache-DiT accelerates Diffusion Transformers by caching intermediate computation results across denoising steps. Adjacent steps produce similar features, so redundant computations can be skipped.

Three caching strategies:
- **DBCache**: Dynamic block-level caching — selectively computes or caches transformer blocks based on residual differences
- **TaylorSeer**: Calibration-based prediction using Taylor expansion to estimate block outputs
- **SCM** (Step Computation Masking): Dynamic step skipping based on configurable policies

**Typical speedup**: 1.5-2.5x depending on model and configuration.

**Official docs**: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/cache_dit

## Architecture

vLLM-Omni integrates cache-dit through `CacheDiTBackend`:

| Component | Purpose |
|-----------|---------|
| `CacheDiTBackend` | Unified backend — auto-selects enabler (standard or custom) |
| `enable_cache_for_dit()` | Default enabler for standard single-transformer models |
| `CUSTOM_DIT_ENABLERS` dict | Registry of custom enablers keyed by pipeline class name |
| `BlockAdapter` | Wraps complex architectures (multi-block-list or multi-transformer) |
| `ForwardPattern` | Specifies block forward signature: `Pattern_0`, `Pattern_1`, `Pattern_2` |
| `ParamsModifier` | Per-transformer or per-block-list config customization |
| `DBCacheConfig` | Configuration for DBCache parameters |
| `cache_dit.refresh_context()` | Updates cache context when `num_inference_steps` changes |

**Source files**:
- `vllm_omni/diffusion/cache/cache_dit_backend.py` — `CacheDiTBackend`, enablers, `CUSTOM_DIT_ENABLERS`
- `vllm_omni/diffusion/cache/` — cache backend implementations

## Standard Models: Automatic Support

Most DiT models follow this pattern:
- Single transformer with one `nn.ModuleList` of blocks
- Standard forward signature
- Compatible with cache-dit's automatic detection

**Examples**: Qwen-Image, Z-Image, FLUX

No code changes needed. `CacheDiTBackend` automatically uses `enable_cache_for_dit()`:

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
    }
)
```

What happens automatically:

```python
def enable_cache_for_dit(pipeline, cache_config):
    db_cache_config = DBCacheConfig(
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )

    cache_dit.enable_cache(pipeline.transformer, cache_config=db_cache_config)

    def refresh_cache_context(pipeline, num_inference_steps, verbose=True):
        cache_dit.refresh_context(
            pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose
        )
    return refresh_cache_context
```

## Custom Architectures: Writing Custom Enablers

### When you need a custom enabler

- Model has multiple block lists in one transformer (e.g., `transformer_blocks` + `single_transformer_blocks`)
- Model has two transformers (e.g., high-noise + low-noise like Wan2.2)
- Model uses non-standard block forward signature

### Pattern 1: Multi-Block-List (LongCat-Image style)

Single transformer with two block lists:

```python
import cache_dit
from cache_dit import BlockAdapter, ForwardPattern, ParamsModifier, DBCacheConfig

def enable_cache_for_your_model(pipeline, cache_config):
    db_cache_config = DBCacheConfig(
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )

    cache_dit.enable_cache(
        BlockAdapter(
            transformer=pipeline.transformer,
            blocks=[
                pipeline.transformer.transformer_blocks,
                pipeline.transformer.single_transformer_blocks,
            ],
            forward_pattern=[ForwardPattern.Pattern_1, ForwardPattern.Pattern_1],
            params_modifiers=[ParamsModifier(...)],
        ),
        cache_config=db_cache_config,
    )

    def refresh_cache_context(pipeline, num_inference_steps, verbose=True):
        cache_dit.refresh_context(
            pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose
        )
    return refresh_cache_context
```

For single transformer with multiple block lists, `refresh_context` works the same as standard models — call it once on the transformer.

### Pattern 2: Dual-Transformer (Wan2.2 style)

Two transformers with separate configs:

```python
def enable_cache_for_dual_transformer(pipeline, cache_config):
    db_cache_config = DBCacheConfig(...)

    cache_dit.enable_cache(
        BlockAdapter(
            transformer=[pipeline.transformer, pipeline.transformer_2],
            blocks=[pipeline.transformer.blocks, pipeline.transformer_2.blocks],
            forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
            params_modifiers=[
                ParamsModifier(...),  # Config for transformer 1
                ParamsModifier(...),  # Config for transformer 2
            ],
        ),
        cache_config=db_cache_config,
    )

    def refresh_cache_context(pipeline, num_inference_steps, verbose=True):
        high_steps, low_steps = _split_inference_steps(num_inference_steps)
        cache_dit.refresh_context(
            pipeline.transformer, num_inference_steps=high_steps, verbose=verbose
        )
        cache_dit.refresh_context(
            pipeline.transformer_2, num_inference_steps=low_steps, verbose=verbose
        )
    return refresh_cache_context
```

Key difference: `refresh_context` must be called on **each transformer separately** with its own step count.

### Choosing the ForwardPattern

| Pattern | Block forward signature | Example models |
|---------|------------------------|----------------|
| `Pattern_0` | `block(hidden_states, **kwargs)` → residual added inside block | Default |
| `Pattern_1` | `block(hidden_states, **kwargs)` → returns `(hidden_states, ...)` tuple | FLUX-style single blocks |
| `Pattern_2` | `block(hidden_states, **kwargs)` → `(hidden_states, ...)` with different residual pattern | Wan2.2 blocks |

Inspect your block's `forward()` return type and residual connection pattern to choose the right one. See [Cache-DiT API Reference](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/) for details.

## Registering Custom Enablers

Add your enabler to `CUSTOM_DIT_ENABLERS` in `vllm_omni/diffusion/cache/cache_dit_backend.py`:

```python
CUSTOM_DIT_ENABLERS = {
    "Wan22Pipeline": enable_cache_for_wan22,
    "LongCatImagePipeline": enable_cache_for_longcat_image,
    "YourModelPipeline": enable_cache_for_your_model,
}
```

The key must match `pipeline.__class__.__name__`.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Fn_compute_blocks` | 1 | Number of blocks to always compute at the front |
| `Bn_compute_blocks` | 0 | Number of blocks to always compute at the back |
| `max_warmup_steps` | 4 | Steps to run without caching at the beginning |
| `max_cached_steps` | — | Max total cached steps |
| `max_continuous_cached_steps` | — | Max consecutive cached steps |
| `residual_diff_threshold` | 0.24 | Threshold for deciding whether to cache a block |

### Tuning for quality vs speed

| Goal | Adjustments |
|------|-------------|
| **More speed, acceptable quality loss** | Higher `residual_diff_threshold` (0.24-0.4), lower `max_warmup_steps` (2-4) |
| **Better quality, less speed** | Lower `residual_diff_threshold` (0.12-0.18), higher `max_warmup_steps` (6-8), lower `max_continuous_cached_steps` (2) |

## Testing

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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

CLI (online serving):

```bash
vllm serve your-model --omni --port 8098 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks": 1, "Bn_compute_blocks": 0, "max_warmup_steps": 4}'
```

**Verification checklist**:
1. Logs show "Cache-dit enabled successfully on xxx"
2. Performance: 1.5-2x speedup vs no cache
3. Quality: compare output with `cache_backend=None`

## Excluded Models

Models listed in `_NO_CACHE_ACCELERATION` in `vllm_omni/diffusion/registry.py` do not support cache-dit (e.g., `NextStep11Pipeline`, `StableDiffusionPipeline`). Check this set before attempting to enable cache-dit.

## Reference Implementations

| Model | Path | Notes |
|-------|------|-------|
| Standard DiT | `cache_dit_backend.py::enable_cache_for_dit` | Default enabler, automatic |
| Wan2.2 | `cache_dit_backend.py::enable_cache_for_wan22` | Dual-transformer, auto-detects mode |
| LongCat | `cache_dit_backend.py::enable_cache_for_longcat_image` | Multi-block-list |
| BAGEL | `cache_dit_backend.py::enable_cache_for_bagel` | Complex omni model |
