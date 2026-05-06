# VAE Patch Parallelism Guide


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

VAE Patch Parallelism distributes the VAE (Variational AutoEncoder) decode/encode computation across multiple GPUs by splitting the latent space into spatial tiles or patches. Each GPU processes a subset of tiles in parallel, significantly reducing peak memory consumption during the VAE decode stage while maintaining output quality.

This is particularly useful for:
- **High-resolution image generation** where VAE decode can become a memory bottleneck
- **Memory-constrained environments** where the VAE decode activation peak exceeds available VRAM
- **Multi-GPU setups** where you want to leverage distributed resources for the VAE stage

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).


VAE Patch Parallelism uses two strategies based on image size:

| Strategy | Use Case | How It Works | Overlap Handling | Output Quality |
|----------|----------|--------------|------------------|----------------|
| **Tiled Decode** | Large images (triggers VAE tiling) | Distributes existing VAE tiling computation across ranks. Each rank decodes a subset of overlapping tiles. | Uses VAE's native `blend_v` and `blend_h` functions to seamlessly merge overlapping regions | Bit-identical (same logic as single-GPU tiling) |
| **Patch Decode** | Small images (no VAE tiling) | Splits latent into spatial patches with halos. Each rank decodes one patch with boundary context. | Halo regions provide edge context; core regions are directly stitched without blending | Near-identical (diff < 0.5%, visually imperceptible) |


VAE Patch Parallelism **reuses the DiT process group** (`dit_group`) and does not initialize a separate ProcessGroup. This means:

- **Shared ranks**: VAE patch parallelism uses the same GPU ranks as DiT parallelism (Tensor Parallel, Sequence Parallel, etc.)
- **Combined usage**: VAE patch parallelism is typically used together with other parallelism methods
- **Configuration alignment**: The `vae_patch_parallel_size` should be no greater than the size of your DiT process group

---

## Quick Start

### Basic Usage

Simplest working example:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

# TP=2 for DiT, VAE patch parallel also uses these 2 GPUs
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=2,          # Enable tensor parallelism for DiT
        vae_patch_parallel_size=2,       # Enable VAE patch parallelism
    ),
    vae_use_tiling=True,  # Required for VAE patch parallelism
)

outputs = omni.generate(
    "a futuristic city at sunset, high resolution, 8k",
    OmniDiffusionSamplingParams(
        num_inference_steps=9,
        height=1152,  # High resolution benefits from VAE patch parallel
        width=1152,
    ),
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/text_to_image/`:

```bash
# Text-to-Image with Z-Image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Tongyi-MAI/Z-Image-Turbo \
    --prompt "a futuristic city at sunset" \
    --height 1152 \
    --width 1152 \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-use-tiling
```

### Online Serving

You can enable VAE patch parallelism in online serving via `--vae-patch-parallel-size`:

```bash
# Text-to-Image with Z-Image, TP=2 + VAE patch parallel=2
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8000 \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-use-tiling
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_patch_parallel_size` | int | 1 | Number of GPUs for VAE patch/tile parallelism. Set to 2 or higher to enable. Should typically match `tensor_parallel_size` as they share the same process group. |

Additional requirements:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_use_tiling` | bool | False | Must be set to `True` when using VAE patch parallelism. |

!!! note "Automatic VAE Tiling"
    When `vae_patch_parallel_size > 1` and the model has a distributed VAE (`DistributedVaeMixin`), the system automatically sets `vae_use_tiling=True` if not already enabled.

---

## Best Practices

### When to Use

**Good for:**

- High-resolution image generation and long video generation
- Memory-constrained setups where VAE decode causes OOM
- Multi-GPU environments

**Not for:**

- Low-resolution images/videos where VAE decode is not a bottleneck
- Single GPU setups should use vae tiling decode, but not parallel vae tiling decode
- Models that do not support vae patch parallel

---

## Troubleshooting

### Common Issue 1: Model Not Support VAE Patch Parallel

**Symptoms**:
```
WARNING: vae_patch_parallel_size=2 is set but VAE patch parallelism is NOT enabled for xxxPipeline; ignoring.
```

**Root Cause**: VAE Patch Parallelism requires the model's VAE to implement `DistributedVaeMixin`. At startup, `vllm_omni/diffusion/registry.py` checks whether the instantiated pipeline has a `.vae` attribute that is an instance of `DistributedVaeMixin`. If it does not, the setting is silently ignored:

```python
vae_pp_size = od_config.parallel_config.vae_patch_parallel_size
is_distributed_vae = hasattr(model, "vae") and isinstance(model.vae, DistributedVaeMixin)
if vae_pp_size > 1 and not is_distributed_vae:
    logger.warning(
        "vae_patch_parallel_size=%d is set but VAE patch parallelism is NOT enabled for %s; ignoring.",
        vae_pp_size,
        od_config.model_class_name,
    )
```

**Solutions**:

1. **Use a supported model** (recommended): check [Supported Models](../../diffusion_features.md#supported-models) for the VAE-Patch-Parallel column.

2. To add support for a new model, implement `DistributedVaeMixin` on its VAE class (contributions are welcome).


### Common Issue 2: `vae_patch_parallel_size` Exceeds DiT Process Group Size

**Symptoms**: Shows warning message, and vae patch parallel size is resized to DiT process group size

**Root Cause**: VAE Patch Parallelism reuses the DiT process group.

**Recommendation**: Always set `vae_patch_parallel_size` to be no greater than your DiT process group size.

Note that the size of DiT process group size equals to:
```text
dit_parallel_size = data_parallel_size
                  × cfg_parallel_size
                  × sequence_parallel_size
                  × pipeline_parallel_size
                  × tensor_parallel_size

```
_sequence_parallel_size = ulysses_degree × ring_degree_

---

## Summary

1. ✅ **Enable VAE Patch Parallelism** - Set `vae_patch_parallel_size`， `vae_use_tiling=True` in `DiffusionParallelConfig` to reduce VAE decode peak memory
2. ✅ **Use Long Sequence** - VAE patch parallelism benefits are most apparent at long sequence decoding
3. ✅ **Combine with other parallelism methods** - Suggest to use together with Tensor Parallel or CFG-Parallel for maximum memory savings
