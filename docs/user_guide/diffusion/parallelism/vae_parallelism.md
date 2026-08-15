# VAE Parallelism Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Independent MiniMax-H3 VAE Groups](#independent-minimax-h3-vae-groups)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

VAE parallelism distributes VAE (Variational AutoEncoder) decode/encode work across multiple GPUs. This guide covers VAE patch/tile parallelism, which splits latent space into spatial tiles or patches, and Wan spatial-shard decode, which shards decoder feature maps along height or width.

This is particularly useful for:
- **High-resolution image generation** where VAE decode can become a memory bottleneck
- **Memory-constrained environments** where the VAE decode activation peak exceeds available VRAM
- **Multi-GPU setups** where you want to leverage distributed resources for the VAE stage

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).


VAE patch parallelism uses two strategies based on image size:

| Strategy | Use Case | How It Works | Overlap Handling | Output Quality |
|----------|----------|--------------|------------------|----------------|
| **Tiled Decode** | Large images (triggers VAE tiling) | Distributes existing VAE tiling computation across ranks. Each rank decodes a subset of overlapping tiles. | Uses VAE's native `blend_v` and `blend_h` functions to seamlessly merge overlapping regions | Bit-identical (same logic as single-GPU tiling) |
| **Patch Decode** | Small images (no VAE tiling) | Splits latent into spatial patches with halos. Each rank decodes one patch with boundary context. | Halo regions provide edge context; core regions are directly stitched without blending | Near-identical (diff < 0.5%, visually imperceptible) |


VAE Patch Parallelism reuses the DiT process group (`dit_group`) by default. MiniMax-H3 is the exception: it can bind its native tile-parallel VAE to an independently sized process group, as described in [Independent MiniMax-H3 VAE Groups](#independent-minimax-h3-vae-groups).

For models using the default behavior:

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
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091 \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-use-tiling
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_patch_parallel_size` | int | 1 | Number of GPUs for VAE patch/tile parallelism. Set to 2 or higher to enable. It should typically match `tensor_parallel_size` for models that reuse the DiT group. For MiniMax-H3, it must not exceed and must evenly divide the DiT process-group size. |
| `vae_parallel_mode` | str | `"tile"` | VAE parallel decode strategy: `"tile"` (default tile/patch parallel decode), `"spatial_shard_height"`, or `"spatial_shard_width"` (spatially-sharded decode, Wan only). See [Spatially-Sharded Decode](#spatially-sharded-decode-wan). |

Additional requirements:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_use_tiling` | bool | False | Must be set to `True` when using VAE patch parallelism. |

!!! note "Automatic VAE Tiling"
    When `vae_patch_parallel_size > 1` and the model has a distributed VAE (`DistributedVaeMixin`), the system automatically sets `vae_use_tiling=True` if not already enabled.

---

## Independent MiniMax-H3 VAE Groups

MiniMax-H3 and its modular pipeline bind native VAE tile parallelism to a
dedicated process group when the requested VAE size is smaller than the DiT
world group. A size of one remains local, while a VAE size equal to the world
continues to reuse the existing DiT group. For example, four diffusion
ranks can be partitioned into the deterministic VAE groups `[0, 1]` and
`[2, 3]`:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
    --omni --trust-remote-code \
    --num-gpus 4 --tensor-parallel-size 2 --usp 2 \
    --vae-patch-parallel-size 2 \
    --vae-parallel-mode tile \
    --vae-use-tiling
```

The following constraints are validated before distributed initialization:

- The worker world size must equal the DiT process-group size. Multiple DiT
  partitions within one worker world are not supported by this path.
- `vae_patch_parallel_size` must be positive, no greater than the DiT
  process-group size, and evenly divide it.
- Only `vae_parallel_mode=tile` is supported for MiniMax-H3. H3 spatial
  sharding remains unsupported.
- Every rank creates every subgroup in the same order. Each rank then retains
  only the subgroup that contains it, avoiding process-group creation order
  mismatches.

The dedicated VAE group is destroyed with the other model-parallel groups.
Other VAE implementations continue to use their existing DiT-group behavior.

---

## Spatially-Sharded Decode (Wan)

The default `vae_parallel_mode="tile"` distributes whole tiles across ranks. For the **Wan** VAE there is an alternative decode strategy, **spatially-sharded decode**, selected via `vae_parallel_mode="spatial_shard_height"` or `vae_parallel_mode="spatial_shard_width"`.

Instead of assigning independent tiles to ranks, spatial-shard decode shards the decoder feature maps along the height (`spatial_shard_height`) or width (`spatial_shard_width`) dimension and exchanges halo rows/columns between neighboring ranks around the spatial convolutions. This keeps the receptive field correct across shard boundaries, so the result matches the single-GPU decode within numerical tolerance.

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=2,
        vae_patch_parallel_size=2,               # must match the DiT group size
        vae_parallel_mode="spatial_shard_width", # or "spatial_shard_height"
    ),
)
```

Or from the CLI / serving entrypoint:

```bash
vllm serve Wan-AI/Wan2.1-T2V-1.3B-Diffusers --omni \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-parallel-mode spatial_shard_width
```

**Constraints and behavior:**

- Spatial-shard decode is **decode-only** and currently implemented for the **Wan** VAE. Other models ignore `spatial_shard_*` modes.
- It requires `vae_patch_parallel_size` to **match the DiT process group size**. If it does not, the VAE logs a warning and **falls back to tile-parallel decode** at runtime.
- `spatial_shard_height` and `spatial_shard_width` are mutually exclusive for a given VAE instance (the decoder is patched in place for a single split dimension).

For end-to-end latency/throughput, launch serving with the desired `vae_parallel_mode` and use the existing diffusion serving benchmark:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --endpoint /v1/videos --dataset random --task t2v --num-prompts 1 \
    --height 480 --width 832 --num-frames 17 --max-concurrency 1
```

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

**Root Cause**: Most VAE implementations reuse the DiT process group.

**Recommendation**: For models that reuse the DiT group, set
`vae_patch_parallel_size` no greater than the DiT process group size. For
MiniMax-H3, choose a tile-mode size that does not exceed and evenly divides the
DiT process-group size; a smaller size uses an independent group.

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
