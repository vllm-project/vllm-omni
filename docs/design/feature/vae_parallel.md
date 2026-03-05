# VAE Patch Parallelism

This document describes how to add **VAE Patch Parallelism** support to a diffusion model.
We use **Qwen-Image** as the reference implementation.

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

### Step 1: Implement DistributedAutoencoderKLQwenImage
`QwenImagePipeline` use `AutoencoderKLQwenImage` for vae, so implement a distributed version:


```
class DistributedAutoencoderKLQwenImage(AutoencoderKLQwenImage, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model
```
**Key points**:
+ Inherit both AutoencoderKLQwenImage and DistributedVaeMixin
+ Call init_distributed() after loading weights

### Step 2: Implement split/exec/merge
Reuse `AutoencoderKLQwenImage.tiled_decode` logic and divide it into three stages. And we need return tiles with `GridSpec` and `TileTask`:
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
    _, _, num_frames, height, width = z.shape
    sample_height = height * self.spatial_compression_ratio
    sample_width = width * self.spatial_compression_ratio

    tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
    tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
    tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
    tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

    blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
    blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

    # Split z into overlapping tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    tiletask_list = []
    for i in range(0, height, tile_latent_stride_height):
        for j in range(0, width, tile_latent_stride_width):
            time_list = []
            for k in range(num_frames):
                self._conv_idx = [0]
                tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                time_list.append(tile)
            tiletask_list.append(
                TileTask(
                    len(tiletask_list),
                    (i // tile_latent_stride_height, j // tile_latent_stride_width),
                    time_list,
                    workload=time_list[0].shape[3] * time_list[0].shape[4],
                )
            )
    tile_spec = {
        "sample_height": sample_height,
        "sample_width": sample_width,
        "blend_height": blend_height,
        "blend_width": blend_width,
    }
    grid_spec = GridSpec(
        split_dims=(3, 4),
        grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
        tile_spec=tile_spec,
        output_dtype=self.dtype,
    )
    return tiletask_list, grid_spec

def tile_exec(self, task: TileTask) -> torch.Tensor:
    """Decode a single latent tile into RGB space."""
    self.clear_cache()
    time = []
    for k in range(len(task.tensor)):
        self._conv_idx = [0]
        tile = self.post_quant_conv(task.tensor[k])
        decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        time.append(decoded)
    result = torch.cat(time, dim=2)
    return result

def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
    """Merge decoded tiles into a full image."""
    grid_h, grid_w = grid_spec.grid_shape
    result_rows = []
    self.clear_cache()

    result_rows = []
    for i in range(grid_h):
        result_row = []
        for j in range(grid_w):
            tile = coord_tensor_map[(i, j)]
            if i > 0:
                tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
            if j > 0:
                tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
            result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
        result_rows.append(torch.cat(result_row, dim=-1))
    dec = torch.cat(result_rows, dim=3)[
        :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
    ]
    return dec
```

### Step 3: Override tiled_decode
We need to override tiled_decode, the main logic is:
+ check distributed is enabled
+ select split/exec/merge
+ Invoke self.distributed_decoder.execute to decode
```
def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
    if not self.is_distributed_enabled():
        return super().tiled_decode(z, return_dict=return_dict)

    logger.info("Decode run with distributed executor")
    result = self.distributed_decoder.execute(
        z,
        DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
        broadcast_result=True,
    )
    if not return_dict:
        return (result,)

    return DecoderOutput(sample=result)
```
`broadcast_result` is set to True or False depending on the model; when enabled, the result will be used even on ranks other than 0.

### Step 4: Modify Pipeline
Change vae model from AutoencoderKLQwenImage to DistributedAutoencoderKLQwenImage
```
class YourModelPipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        ...
-       self.vae = AutoencoderKL.from_pretrained(
-           model, subfolder="vae", local_files_only=local_files_only).to(self.device)
+       self.vae = DistributedAutoencoderKL.from_pretrained(
+           model, subfolder="vae", local_files_only=local_files_only
+       ).to(self.device)
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
            vae_patch_parallel_size=1, # or 2
        ),
    )
```
When vae_patch_parallel_size is larger than the DiT world size, it will automatically fall back to using the DiT world size instead.

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Notes |
|-------|------|-------|
| **Z-Image** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl.py` | Distributed AutoencoderKL |
| **Wan2.2** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_wan.py` | Distributed AutoencoderKLWan |
| **Qwen-Image** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_qwenimage.py` | Distributed AutoencoderKLQwenImage |

---

## Summary

Adding Vae Patch Parallel support to diffusion model:

1. **Implement Distributed Vae** - mainly copy from `diffusers` tiled_decode, and refactor into split/exec/merge
2. **Change vae model in pipeline to Distributed Vae**
3. **Test** - Verify with `tensor_parallel_size=N` quality
