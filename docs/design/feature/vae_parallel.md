# VAE Parallel

This document defines the VAE parallelism strategy contract and describes how to
add a strategy to a diffusion model. Tile/patch parallelism is the portable
baseline. Spatial-shard parallelism is a model-specific decode strategy that can
be selected explicitly or through a request-aware policy.

We use **Qwen-Image** as the reference implementation for tile-parallel decode,
**Wan2.2** for tile-parallel encode, and **Wan2.2** as the first spatial-shard
decode implementation. Other VAE families can add spatial strategies later
without changing the public mode meanings defined here.

---

## Table of Contents

- [Overview](#overview)
- [Strategy Selection Contract](#strategy-selection-contract)
- [Tile/Patch Implementation (Decode)](#tilepatch-implementation-decode)
- [Tile/Patch Encode Implementation](#tilepatch-encode-implementation)
- [Testing](#testing)
- [Reference Implementations](#reference-implementations)
- [Spatially-Sharded Decode](#spatially-sharded-decode)
- [Adding Another VAE Strategy](#adding-another-vae-strategy)
- [Summary](#summary)

---

## Overview

### What is VAE parallelism?

VAE parallelism distributes VAE **encoding** or **decoding** across multiple
ranks. The implementation currently has two strategy families:

| Strategy | Unit of work | Communication | Current scope |
|----------|--------------|---------------|---------------|
| Tile/patch | Independent overlapping tiles | Gather, stitch, and optionally broadcast | Encode and decode across supported distributed VAEs |
| Spatial shard | One feature map sharded along a spatial axis | Per-layer halo exchange followed by output gather | Wan decode |

Both strategies preserve the model-specific receptive field and reconstruct the
same logical output as single-rank execution. They differ in how work is divided:
tile parallelism assigns complete overlapping tiles to ranks, while spatial
sharding keeps one logical feature map partitioned throughout the decoder.

This approach can:

+ Distribute computation across multiple devices
+ Reduce peak memory usage per device
+ Accelerate encoding or decoding latency

The actual benefit depends on shape, topology, communication cost, and VAE
architecture. A strategy must not become the global default without parity and
performance evidence on its supported models.

### When to Use Encode vs Decode Parallel

| Operation | Use Case | Example |
|-----------|----------|---------|
| **Decode Parallel** | Text-to-Image, Text-to-Video | Latent → Image/Video |
| **Encode Parallel** | Image-to-Video (I2V) | Image → Latent (for conditioning) |

### Architecture and ownership

`DiffusionParallelConfig.vae_parallel_mode` carries the public strategy request.
The shared registry forwards that value to the VAE, but each distributed VAE
adapter owns its supported modes, request-shape selection, topology validation,
and fallback behavior. Model-specific policy must not be added to the shared
runner or registry.

**DistributedVaeExecutor** is the model-agnostic core for tile-parallel VAE
encoding and decoding.

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

Spatial-shard implementations use the same configured VAE process group but do
not use the tile task executor. They keep the feature map sharded across decoder
layers and own their halo-exchange and final-gather contracts inside the model's
distributed autoencoder implementation.

#### Why split / exec / merge is necessary?

The latent tensor cannot be arbitrarily partitioned.

During decoding:

+ Each output pixel may depend on neighboring pixels

+ The receptive field is model-dependent

Therefore:

+ Tiles must include overlap

+ Merge must perform blending to avoid seams

## Strategy Selection Contract

`vae_parallel_mode="tile"` is the compatibility-preserving global default.
Other values are stable strategy requests, but support remains model-specific.

| Mode | Meaning | Current implementation | Unsupported or ineligible request |
|------|---------|------------------------|-----------------------------------|
| `tile` | Use the existing tile/patch executor | All distributed VAE adapters | Not applicable |
| `auto` | Let the VAE adapter select a validated strategy for each request | Wan decode selects height or width spatial sharding | Wan uses tile when ineligible; unsupported VAE families reject or document their fallback |
| `spatial_shard_height` | Force spatial-shard decode along height | Wan decode | Wan warns and uses tile when ineligible; unsupported VAE families reject |
| `spatial_shard_width` | Force spatial-shard decode along width | Wan decode | Wan warns and uses tile when ineligible; unsupported VAE families reject |

The shared configuration layer validates the mode name. It does not decide
whether a model supports that mode. A VAE adapter adding a mode must:

1. Keep `tile` as an exact feature-off path.
2. Select a strategy per request rather than from stale process-global state.
3. Validate input rank/shape and process-group membership before collectives.
4. Reject or explicitly fall back from unsupported combinations; never silently
   reinterpret one forced strategy as another.
5. Preserve direct encode/decode behavior when Diffusers does not select a tiled
   path.

`auto` is a policy hook, not a universal heuristic. Each VAE family may select
from only the strategies it has validated. Adding another family must not change
Wan's selector or the meanings of the explicit modes.

## Tile/Patch Implementation (Decode)

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
+ Invoke self.distributed_executor.execute to decode
```
def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
    if not self.is_distributed_enabled():
        return super().tiled_decode(z, return_dict=return_dict)

    logger.info("Decode run with distributed executor")
    result = self.distributed_executor.execute(
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

## Tile/Patch Encode Implementation

For models that require VAE encoding (e.g., Image-to-Video), you can also parallelize the encode operation. We use **Wan2.2** as the reference implementation.

### Step 1: Implement encode_tile_split

Similar to decode, split the input tensor into tiles. Key considerations:

+ **Patchify handling**: If the model uses `patch_size`, scale tile parameters accordingly
+ **Temporal chunking**: Video VAEs may have temporal compression (e.g., 4x)

```python
def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
    _, _, num_frames, height, width = x.shape
    encode_spatial_compression_ratio = self.spatial_compression_ratio

    # Scale tile parameters for patchified coordinate system
    tile_sample_min_height = self.tile_sample_min_height
    tile_sample_min_width = self.tile_sample_min_width
    tile_sample_stride_height = self.tile_sample_stride_height
    tile_sample_stride_width = self.tile_sample_stride_width

    if self.config.patch_size is not None:
        # When input is patchified, scale tile parameters accordingly
        encode_spatial_compression_ratio = self.spatial_compression_ratio // self.config.patch_size
        tile_sample_min_height = tile_sample_min_height // self.config.patch_size
        tile_sample_min_width = tile_sample_min_width // self.config.patch_size
        tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
        tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size

    latent_height = height // encode_spatial_compression_ratio
    latent_width = width // encode_spatial_compression_ratio

    tile_latent_min_height = tile_sample_min_height // encode_spatial_compression_ratio
    tile_latent_min_width = tile_sample_min_width // encode_spatial_compression_ratio
    tile_latent_stride_height = tile_sample_stride_height // encode_spatial_compression_ratio
    tile_latent_stride_width = tile_sample_stride_width // encode_spatial_compression_ratio

    blend_height = tile_latent_min_height - tile_latent_stride_height
    blend_width = tile_latent_min_width - tile_latent_stride_width

    tiletask_list = []
    # Use temporal compression ratio from config instead of hardcoding
    temporal_compression = self.config.scale_factor_temporal

    for i in range(0, height, tile_sample_stride_height):
        for j in range(0, width, tile_sample_stride_width):
            time_list = []
            frame_range = 1 + (num_frames - 1) // temporal_compression
            for k in range(frame_range):
                if k == 0:
                    tile = x[:, :, :1, i : i + tile_sample_min_height, j : j + tile_sample_min_width]
                else:
                    tile = x[
                        :, :,
                        1 + temporal_compression * (k - 1) : 1 + temporal_compression * k,
                        i : i + tile_sample_min_height,
                        j : j + tile_sample_min_width,
                    ]
                time_list.append(tile)
            tiletask_list.append(
                TileTask(len(tiletask_list), (i // tile_sample_stride_height, j // tile_sample_stride_width),
                         time_list, workload=time_list[0].shape[3] * time_list[0].shape[4])
            )

    grid_spec = GridSpec(
        split_dims=(3, 4),
        grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
        tile_spec={
            "latent_height": latent_height, "latent_width": latent_width,
            "blend_height": blend_height, "blend_width": blend_width,
            "tile_latent_stride_height": tile_latent_stride_height,
            "tile_latent_stride_width": tile_latent_stride_width,
        },
        output_dtype=self.dtype,
    )
    return tiletask_list, grid_spec
```

### Step 2: Implement encode_tile_exec

```python
def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
    """Encode a single sample tile into latent space."""
    self.clear_cache()
    time = []
    for k, tile in enumerate(task.tensor):
        self._enc_conv_idx = [0]
        encoded = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        encoded = self.quant_conv(encoded)
        time.append(encoded)
    result = torch.cat(time, dim=2)
    self.clear_cache()
    return result
```

### Step 3: Implement encode_tile_merge

```python
def encode_tile_merge(
    self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
) -> torch.Tensor:
    """Merge encoded tiles into a full latent tensor."""
    grid_h, grid_w = grid_spec.grid_shape
    result_rows = []
    for i in range(grid_h):
        result_row = []
        for j in range(grid_w):
            tile = coord_tensor_map[(i, j)]
            if i > 0:
                tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
            if j > 0:
                tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
            result_row.append(tile[:, :, :,
                : grid_spec.tile_spec["tile_latent_stride_height"],
                : grid_spec.tile_spec["tile_latent_stride_width"]])
        result_rows.append(torch.cat(result_row, dim=-1))

    enc = torch.cat(result_rows, dim=3)[
        :, :, :, : grid_spec.tile_spec["latent_height"], : grid_spec.tile_spec["latent_width"]
    ]
    return enc
```

### Step 4: Override tiled_encode method

Override `tiled_encode` instead of `encode`. The parent's `_encode()` handles patchify before calling `tiled_encode()`, so input `x` is already patchified.

```python
def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
    """
    Encode using distributed VAE executor.

    Note: x is already patchified by parent's _encode() before calling this method.
    """
    if not self.is_distributed_enabled():
        return super().tiled_encode(x)

    self.clear_cache()
    result = self.distributed_executor.execute(
        x,
        DistributedOperator(
            split=self.encode_tile_split,
            exec=self.encode_tile_exec,
            merge=self.encode_tile_merge,
        ),
        broadcast_result=True,  # Latents needed by all ranks for diffusion
    )
    self.clear_cache()
    return result
```

**Key differences from decode parallel:**

| Aspect | Decode Parallel | Encode Parallel |
|--------|-----------------|-----------------|
| `broadcast_result` | Often `False` (only rank 0 needs output) | `True` (all ranks need latents for diffusion) |
| Patchify | Applied in merge (unpatchify) | Handled by parent `_encode()` before `tiled_encode()` |
| Temporal chunking | Frame-by-frame | Chunk-based (e.g., 1 + 4n frames) |

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

| Model | Path | Decode Parallel | Encode Parallel |
|-------|------|-----------------|-----------------|
| **Z-Image** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl.py` | ✅ | ❌ |
| **Wan2.2** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_wan.py` | ✅ | ✅ |
| **Qwen-Image** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_qwenimage.py` | ✅ | ❌ |
| **FLUX.2-dev** | `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_flux2.py` | ✅ | ✅ |

---

## Spatially-Sharded Decode

The tile-parallel executor above assigns independent spatial **tiles** to ranks.
A spatial-shard backend instead partitions one global decoder feature map along
height or width. Wan is the first supported implementation.

`DiffusionParallelConfig.vae_parallel_mode="tile"` remains the default. For Wan,
opt into request-shape selection with `"auto"`, or force an axis with
`"spatial_shard_height"` or `"spatial_shard_width"`.

### How it differs from tile parallel

| Aspect | Tile parallel (`"tile"`) | Automatic (`"auto"`) | Spatially-sharded (`"spatial_shard_height"`/`"spatial_shard_width"`) |
|--------|--------------------------|----------------------|-----------------------------------------------|
| Unit of work | Independent overlapping tiles | Chooses tile or a spatial shard per request | A single global feature map sharded along H or W |
| Cross-rank communication | Gather tiles to rank 0, stitch + blend | Depends on the selected path | Per-conv **halo exchange** of boundary rows/cols (P2P) |
| Output assembly | Blend overlapping tiles | Depends on the selected path | All-gather shards on rank 0, trim padding (matches `broadcast_result=False`) |
| Scope | Decode + encode | Wan decode selection; encode remains tiled | Wan decode only |

Spatial-shard decode swaps the decoder's spatial convolutions/padding for halo-exchanging variants (`WanDistConv2d`, `WanDistCausalConv3d`, `WanDistZeroPad2d`) so each rank only holds a shard of the activations but still sees the correct receptive field at shard boundaries. Implementation lives in `vllm_omni/diffusion/distributed/autoencoders/wan_spatial_shard.py`.

### Wiring

`vae_parallel_mode` flows through the same path as `vae_patch_parallel_size`:

```text
serve.py (--vae-parallel-mode) / OmniEngineArgs
  -> DiffusionParallelConfig.vae_parallel_mode
  -> registry.py: model.vae.set_parallel_size(vae_pp_size, mode=...)
  -> DistributedVaeExecutor.parallel_mode
  -> DistributedAutoencoderKLWan.tiled_decode dispatch
```

`DistributedAutoencoderKLWan._spatial_shard_decode_enabled()` gates the path: it
requires distributed decode to be enabled, a 5D latent, and
`vae_patch_parallel_size == DiT group size`. The method is reached only after
Diffusers has selected tiled decode for the request. In `auto` mode the longer
axis that can cover every rank is selected, with width winning ties. If neither
axis can cover the group, auto mode uses tile decode. This policy is cached only
through normal kernel shape caches; it is reevaluated for every request.

### Notes

- The decoder is patched **in place** the first time spatial-shard decode runs. Its wrappers consult a context-local split dimension and retain an exact direct fallback, allowing later requests to select tile, height, or width safely.
- Numerical correctness vs. single-GPU decode is covered by `tests/diffusion/distributed/test_wan_spatial_shard.py::test_spatial_shard_decode_matches_reference` (multi-GPU, nightly `full_model` + `distributed_cuda`).

## Adding Another VAE Strategy

Add future strategies through the distributed autoencoder that owns the model
semantics. Do not add model-name dispatch to the shared runner, registry, or
`DistributedVaeExecutor`.

For a new VAE family:

1. **Declare capability**: document which modes, operations, shapes, dtypes, and
   process-group layouts the adapter supports.
2. **Implement model-local selection**: keep `auto` policy and explicit-mode
   dispatch in the distributed autoencoder. The selector may use request shape,
   but must be deterministic and side-effect free.
3. **Preserve the baseline**: retain exact tile/direct fallbacks and state-dict
   compatibility. Structural wrappers must not register duplicate parameters or
   alter checkpoint names.
4. **Define communication**: derive shards from the configured VAE group, keep
   collectives symmetric, specify padding/halo ownership, and return the same
   output placement as the tile path.
5. **Bound request state**: make the selected axis and scratch buffers
   request-scoped, release temporary buffers on success and failure, and support
   alternating modes in one long-lived worker.
6. **Validate independently**: compare each explicit strategy against the
   single-rank reference, then test `auto`, tile fallback, unsupported topology,
   and a mixed request sequence. Record per-rank peak memory and decode latency
   before documenting a preferred mode.

Promote a selector into shared infrastructure only after at least two VAE
families use the same inputs, invariants, fallback behavior, and lifecycle. Until
then, shared code owns transport and configuration while each model adapter owns
policy.

---

## Summary

Adding VAE parallel support to a diffusion model:

1. **Implement the distributed VAE** - Inherit from the base VAE class and
   `DistributedVaeMixin`.
2. **Keep tile as the baseline** - For tile decode, refactor `tiled_decode` into
   `tile_split`/`tile_exec`/`tile_merge`.
3. **Add encode parallelism when needed** - Implement
   `encode_tile_split`/`encode_tile_exec`/`encode_tile_merge` for I2V models.
4. **Add model-specific strategies locally** - Own selectors, topology checks,
   communication, and fallback in the distributed autoencoder.
5. **Wire the pipeline** - Replace the pipeline VAE with its distributed version.
6. **Test strategy and lifecycle parity** - Compare one rank with every supported
   strategy, including feature-off, fallback, and mixed-request sequences.
