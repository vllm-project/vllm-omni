# TeaCache Configuration Guide

TeaCache speeds up diffusion model inference by caching transformer computations when consecutive timesteps are similar. This typically provides **1.5x-2.0x speedup** with minimal quality loss.

## Quick Start

Enable TeaCache by setting `cache_backend` to `"tea_cache"`:

```python
from vllm_omni import Omni

# Simple configuration - model_type is automatically extracted from pipeline.__class__.__name__
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2  # Optional, defaults to 0.2
    }
)
outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
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
    cache_config={"rel_l1_thresh": 0.2}  # Optional
)
```

## Online Serving (OpenAI-Compatible)

Enable TeaCache for online serving by passing `--cache-backend tea_cache` when starting the server:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.2}'
```

## Configuration Parameters

### `rel_l1_thresh` (float, default: `0.2`)

Controls the balance between speed and quality. Lower values prioritize quality, higher values prioritize speed.

**Recommended values:**

- `0.2` - **~1.5x speedup** with minimal quality loss (recommended)
- `0.4` - **~1.8x speedup** with slight quality loss
- `0.6` - **~2.0x speedup** with noticeable quality loss
- `0.8` - **~2.25x speedup** with significant quality loss

## Examples

### Python API

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}
)
outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```

## Performance Tuning

Start with the default `rel_l1_thresh=0.2` and adjust based on your needs:

- **Maximum quality**: Use `0.1-0.2`
- **Balanced**: Use `0.2-0.4` (recommended)
- **Maximum speed**: Use `0.6-0.8` (may reduce quality)

## Troubleshooting

### Quality Degradation

If you notice quality issues, lower the threshold:

```python
cache_config={"rel_l1_thresh": 0.1}  # More conservative caching
```

## Supported Models

### ImageGen

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models |
|--------------|--------|-------------------|
| `QwenImagePipeline` | Qwen-Image | `Qwen/Qwen-Image` |
| `QwenImageEditPipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` |
| `QwenImageEditPlusPipeline` | Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` |
| `QwenImageLayeredPipeline` | Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` |
| `BagelForConditionalGeneration` | BAGEL (DiT-only) | `ByteDance-Seed/BAGEL-7B-MoT` |

### VideoGen

No VideoGen models are supported by TeaCache yet.

### Coming Soon

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| Architecture | Models | Example HF Models |
|--------------|--------|-------------------|
| `FluxPipeline` | Flux | - |
| `CogVideoXPipeline` | CogVideoX | - |
