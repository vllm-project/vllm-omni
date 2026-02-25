# VAE Patch Parallelism

This document describes how to add **VAE Patch Parallelism** support to a diffusion model.
We use **Z-Image** as the reference implementation.

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Testing](#testing)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview

### What is Vae Patch parallel?

**VAE Patch Parallelism** is a decoding acceleration technique. Instead of decoding the entire latent tensor at once, the latent tensor is:

+ Split into multiple spatial tiles

+ Distributed across multiple ranks

+ Decoded in parallel

+ Merged to reconstruct the final output

This approach:

+ Distributes computation across multiple devices

+ Reduces peak memory usage per device

+ Accelerates decoding latency

### Architecture
We introduce **DistributedVaeExecutor** as the core component responsible for distributed VAE decoding.

The executor is model-agnostic and accepts three function parameters:

+ split – Partition the latent into tiles

+ exec – Decode a single tile

+ merge – Combine decoded tiles into the final output

#### Execution Flow

+ Call split(z) to generate a list of TileTask and a GridSpec

+ Dispatch tasks across ranks using workload-based balancing

+ Each rank executes exec(task) on its assigned tiles

+ Gather decoded tile results to rank 0

+ Rank 0 performs merge(...)

+ (Optional) Broadcast final result to all ranks

This design separates:

+ Distributed execution logic

+ Model-specific tiling and merging logic

#### Why split / exec / merge is necessary?

The latent tensor cannot be arbitrarily partitioned.

During decoding:

+ Each output pixel may depend on neighboring pixels

+ The receptive field is model-dependent

Therefore:

+ Tiles must include overlap

+ Merge must perform blending to avoid seams

## Step-by-Step Implementation

### Step 1: Implement DistributedAutoencoderKL
`ZImagePipeline` use `AutoencoderKL` for vae, so implement a distributed version:


```
class DistributedAutoencoderKL(AutoencoderKL, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model
```
**Key points**:
+ Inherit both AutoencoderKL and DistributedVaeMixin
+ Call init_distributed() after loading weights

### Step 2: Implement split/exec/merge
Reuse `AutoencoderKL.tiled_decode` logic and divide it into three stages. And we need return tiles with `GridSpec` and `TileTask`:
```
class GridSpec:
    split_dims: tuple[int, ...]  # Tensor dimensions being split (e.g., (2, 3) for (B, C, H, W))
    grid_shape: tuple[int, ...]  # Tile grid layout (num_rows, num_cols)
    tile_spec: dict = field(default_factory=dict) # Metadata required for merging
    output_dtype: torch.dtype | None = None # Final output dtype
```
```
class TileTask:
    tile_id: int # task id
    grid_coord: tuple[int, ...]  # Tile position in grid
    tensor: torch.Tensor | list[torch.Tensor]  # The tile tensor
    workload: int | float = 1 # Used for load balancing (e.g., tile area)
```
And tiled base split/exec/merge as follow:
```
def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
    # mostly copy from AutoencoderKL
    overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
    blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
    row_limit = self.tile_sample_min_size - blend_extent

    # Split z into overlapping 64x64 tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    tiletask_list = []
    for i in range(0, z.shape[2], overlap_size):
        for j in range(0, z.shape[3], overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
            tiletask_list.append(
                TileTask(
                    len(tiletask_list),
                    (i // overlap_size, j // overlap_size),
                    tile,
                    workload=tile.shape[2] * tile.shape[3],
                )
            )

    tile_spec = {
        "blend_extent": blend_extent,
        "row_limit": row_limit,
    }
    grid_spec = GridSpec(
        split_dims=(2, 3),
        grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
        tile_spec=tile_spec,
    )
    return tiletask_list, grid_spec

def tile_exec(self, task: TileTask) -> torch.Tensor:
    """Decode a single latent tile into RGB space."""
    tile = task.tensor
    if self.config.use_post_quant_conv:
        tile = self.post_quant_conv(tile)
    decoded = self.decoder(tile)
    return decoded

def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
    """Merge decoded tiles into a full image."""

    grid_h, grid_w = grid_spec.grid_shape
    result_rows = []
    for i in range(grid_h):
        result_row = []
        for j in range(grid_w):
            tile = coord_tensor_map[(i, j)]
            if i > 0:
                tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_extent"])
            if j > 0:
                tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_extent"])
            result_row.append(tile[:, :, : grid_spec.tile_spec["row_limit"], : grid_spec.tile_spec["row_limit"]])
        result_rows.append(torch.cat(result_row, dim=3))

    dec = torch.cat(result_rows, dim=2)
    return dec
```

### Step 3: Override decode
We need to override decode, the main logic is:
+ check distributed is enabled
+ select split/exec/merge
+ Invoke self.distributed_decoder.execute to decode
```
def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
    if not self.is_distributed_enabled():
        return super().decode(z, return_dict=return_dict, *args, **kwargs)

    split, exec, merge = self._strategy_select(z)

    if split is not None:
        strategy = "tile" if split == self.tile_split else "patch"
        logger.info(f"Decode run with distributed executor, split strategy is {strategy}")
        result = self.distributed_decoder.execute(
            z, DistributedOperator(split=split, exec=exec, merge=merge), broadcast_result=False
        )
        if not return_dict:
            return (result,)

        from diffusers.models.autoencoders.vae import DecoderOutput

        return DecoderOutput(sample=result)
    else:
        return super().decode(z, return_dict=return_dict, *args, **kwargs)
```

### Step 4: Modify Pipeline
Change vae model from AutoencoderKL to DistributedAutoencoderKL
```
self.vae = DistributedAutoencoderKL.from_pretrained(
        model, subfolder="vae", local_files_only=local_files_only
    ).to(self._execution_device)
```

## Testing
Verify numerical consistency between:
+ vae_patch_parallel_size = 1

+ vae_patch_parallel_size = N

Example:
torch.allclose(output_1, output_n, atol=1e-5)

Testing requirements:
+ Fix random seed
+ Use identical tiling strategy

```python
m = Omni(
        model=model_name,
        vae_use_tiling=True,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2,
            vae_patch_parallel_size=1, # or 2, 4
        ),
    )
```

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Notes |
|-------|------|-------|
| **Z-Image** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl.py` | Distributed AutoencoderKL |
| **Wan2.2** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_wan.py` | Distributed AutoencoderKLWan |

---

## Summary

Adding Vae Patch Parallel support to diffusion model:

1. **Implement Distributed Vae** - mainly copy from `diffusers` tiled_decode, and refactor into split/exec/merge
2. **Change vae model in pipeline to Distributed Vae**
3. **Test** - Verify with `tensor_parallel_size=N` quality
