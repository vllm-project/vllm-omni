# PipeFusion

This section describes how to add PipeFusion to a diffusion pipeline. PipeFusion builds on Pipeline Parallelism (PP) by
splitting each denoising sequence into patches during the later denoising steps. The patch loop lets PP stages overlap
communication from one patch with compute from the next patch, reducing pipeline bubbles for supported diffusion
models.

## Implementation Checklist

Adding PipeFusion support requires:

1. Add pipeline warmup and async patch loop support by placing `PipeFusionPipelineMixin` before `PipelineParallelMixin` in the pipeline class
2. Implement `prepare_model_kwargs()` to handle `latents=None` on non-first PP stages
3. Add scheduler patch-cache support with `PipeFusionSchedulerMixin`
4. Add transformer patch support (RoPE slicing, KV-cache updates, unpatchify) with `PipeFusionTransformerMixin`, `PipeFusionRotaryEmbeddingMixin`, and `PipeFusionSelfAttentionMixin`
5. Add Conv3d boundary caching for overlapping convolutions with `Conv3dLayer`

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)

---

## Overview

### What is PipeFusion?

Pipeline Parallelism splits the denoising transformer by layer across sequential PP ranks. PipeFusion keeps that PP
layout, but during the async phase it also splits the current latent sequence into `pipeline_parallel_size` patches.

For each timestep after warmup:

1. The first PP rank splits the current latents into patches.
2. Each patch runs through the normal PP `predict_noise_maybe_with_cfg()` path.
3. The last PP rank runs the scheduler for each patch and sends updated patch latents back to rank 0.
4. Rank 0 reassembles the patches after the async timesteps finish.

The first `warmup_steps` use the standard PP denoising loop. This fills attention KV caches and scheduler state
with full-sequence tensors before the patch loop starts.

### Architecture

PipeFusion is a set of small mixins layered on top of existing PP and CFG abstractions.

| Component                        | Purpose                                                                                               |
|----------------------------------|-------------------------------------------------------------------------------------------------------|
| `PipeFusionPipelineMixin`        | Wraps `forward()`, `diffuse()`, and `prepare_model_kwargs()` for warmup plus async patch execution    |
| `PipeFusionRuntime`              | Tracks patch mode, current patch index, split dimension, token ranges, and communication-buffer reset |
| `PipeFusionSchedulerMixin`       | Splits scheduler caches by patch and gates shared scheduler state updates                             |
| `PipeFusionTransformerMixin`     | Provides patch-aware output shape helpers and unpatchify                                              |
| `PipeFusionRotaryEmbeddingMixin` | Slices full RoPE embeddings to the current patch                                                      |
| `PipeFusionSelfAttentionMixin`   | Maintains full K/V caches while updating the current patch slice                                      |
| `PipeFusionConvMixin`            | Caches activations for overlapping Conv3d layers at patch boundaries                                  |
| `PipelineParallelMixin`          | Provides the inter-stage communication PipeFusion relies on                                           |

### Execution Flow

```mermaid
flowchart LR
    request[Request] --> config[Set PipeFusion Run Config]
    config --> input[Set Input Metadata]
    input --> warmup[Warmup: Standard PP Denoising]
    warmup --> asyncPhase[Async Patch Mode]
    asyncPhase --> split[Split Latents]
    split --> patchLoop[Per-Patch PP Predict And Scheduler]
    patchLoop --> sync[Drain PP Sends]
    sync --> concat[Concatenate Patches]
```

PipeFusion reuses the standard denoising contract:

- `predict_noise_maybe_with_cfg()` still owns PP forward communication.
- `scheduler_step_maybe_with_cfg()` still owns last-rank scheduler execution and last-to-first loopback.
- `PipeFusionPipelineMixin` only changes how often these helpers are called and which communication labels are used.

### Patch Split Dimensions

PipeFusion can split latents along either height or temporal dimensions.

| `split_dim` | Latent split dimension    | Token layout behavior                                                           |
|-------------|---------------------------|---------------------------------------------------------------------------------|
| `height`    | `-2` in `[B, C, T, H, W]` | Patch tokens are interleaved by frame and must be written through a 5D K/V view |
| `temporal`  | `-3` in `[B, C, T, H, W]` | Patch tokens are contiguous in flattened `[frame, height, width]` order         |

The number of patches equals the PP world size. Any remainder is assigned to the last patch.

---

## Step-by-Step Implementation

### Step 1: Add `PipeFusionPipelineMixin`

`PipeFusionPipelineMixin` requires `PipelineParallelMixin` and must appear before it in the class MRO. The mixin enforces
this when the class is defined.

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipefusion.pipefusion import PipeFusionPipelineMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin


class YourPipeline(
    nn.Module,
    PipeFusionPipelineMixin,
    PipelineParallelMixin,
    CFGParallelMixin,
):
    ...
```

The order matters because PipeFusion wraps the pipeline boundary, then delegates prediction and scheduler communication
to the PP-aware methods. The mixin automatically handles the split between warmup and async denoising phases, as well as
the labeled communication using `inter_comm_ids` and `loopback_comm_id`.

### Step 2: Make `prepare_model_kwargs()` patch-aware

Only the first PP stage consumes latent tensors in patch mode. Later stages receive intermediate tensors from upstream PP
ranks, so `PipeFusionPipelineMixin` passes `latents=None` to `prepare_model_kwargs()` on non-first stages.

Model-specific implementations should:

- Preserve full original dimensions in `dims` or an equivalent field.
- Build patch-local latent inputs only when `latents` is not `None`.
- Keep CFG positive and negative kwargs identical to the non-PipeFusion path.

### Step 3: Add scheduler patch caches

Schedulers that keep state across steps should inherit `PipeFusionSchedulerMixin` and declare which attributes are
per-patch caches via `_pipefusion_patch_cache_spec`.

```python
class YourScheduler(PipeFusionSchedulerMixin, BaseScheduler):
    _pipefusion_patch_cache_spec = [
        ("model_outputs", "list"),
        ("last_sample", "tensor"),
    ]
```

The mixin automates the transition from full-sequence to patch-based scheduling:

- **Initial State Splitting**: When async mode starts, it takes the tensors from the warmup phase and splits them into per-patch caches.
- **Automatic State Management**: It swaps the correct patch state into the scheduler before each `step()` and saves the results back into the patch cache after.
- **Shared State Protection**: Prevents internal scheduler variables (like `_step_index` or step counters) from advancing multiple times per timestep. These values are only allowed to increment after the last patch is processed.

### Step 4: Add transformer patch support

PipeFusion transformer support requires integrating several mixins into the model components.

#### 4.1 Inherit `PipeFusionTransformerMixin`

The transformer model must inherit from `PipeFusionTransformerMixin` and implement the `_unpatchify` abstract method.
The mixin automatically wraps `_unpatchify` to handle patch-local dimensions during async mode.

```python
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_transformer import PipeFusionTransformerMixin

class YourTransformer(PipeFusionTransformerMixin, YourBaseModel):
    def _unpatchify(self, hidden_states, dims):
        # REQUIRED: Implementation that reshapes tokens back to original latent space.
        # The mixin ensures this is called with patch-local dimensions
        # during async mode.
        ...
```

#### 4.2 Slice RoPE with `PipeFusionRotaryEmbeddingMixin`

The rotary embedding module should inherit from `PipeFusionRotaryEmbeddingMixin`. This API automatically handles
slicing the full-sequence RoPE embeddings to match the current patch during async mode.

```python
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_transformer import PipeFusionRotaryEmbeddingMixin

class YourRotaryPosEmbed(PipeFusionRotaryEmbeddingMixin, YourBaseRoPE):
    ...
```

#### 4.3 Update full K/V caches with `PipeFusionSelfAttentionMixin`

The attention module should inherit from `PipeFusionSelfAttentionMixin`. This API maintains separate full K/V caches
for conditional and unconditional branches. During the async patch loop, it ensures that each patch's newly computed
keys and values are written into their correct spatial or temporal position within the full-sequence cache, allowing
each patch to attend to the entire sequence.

```python
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_transformer import PipeFusionSelfAttentionMixin

class YourSelfAttention(PipeFusionSelfAttentionMixin, YourBaseAttention):
    ...
```

### Step 5: Add Conv3d boundary caching

For convolutions with `kernel_size != stride` (overlapping convolutions), use `Conv3dLayer` from
`vllm_omni.diffusion.distributed.pipefusion.pipefusion_conv`. This layer stores a full activation cache and performs
sliced convolutions to handle patch boundaries correctly.

When calling the forward pass of these layers during patch mode, you must pass the original full input dimensions.

```python
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_conv import Conv3dLayer

class YourTransformer(nn.Module):
    def __init__(self):
        ...
        self.patch_embedding = Conv3dLayer(...)

    def forward(self, hidden_states, dims=None):
        # hidden_states is the current patch input [B, C, T_patch, H_patch, W_patch]
        # dims is the original full input dimensions [B, C, T, H, W]
        hidden_states = self.patch_embedding(hidden_states, dims=dims)
        ...
```

---

## Testing

PipeFusion testing consists of unit tests for the bookkeeping logic and manual inference checks for end-to-end parity.

### PipeFusion unit tests

Most PipeFusion bookkeeping can be tested on CPU:

- runtime patch metadata for height and temporal splits
- scheduler patch-cache splitting and state restoration
- RoPE slicing
- height and temporal K/V cache writes
- patch-aware unpatchify shape behavior
- Conv3d activation cache behavior
- mixin MRO and request config validation

### Manual inference

Use an offline inference script with PP and PipeFusion enabled:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp4_pf.mp4 \
--pipeline-parallel-size=4 \
--enable-pipefusion \
--pipefusion-warmup-steps=5 \
--pipefusion-split-dim=temporal \
--vae-patch-parallel-size=4
```

Verify that:

1. The run completes without hangs.
2. Output quality remains acceptable for the target workload compared with the standard PP baseline.
3. PP sends are drained before VAE decode.
4. Both `height` and `temporal` split dimensions work for supported shapes.

---

## Troubleshooting

### Issue: PipeFusion fails with `pipeline_parallel_size=1`

**Cause:** PipeFusion relies on PP communication and patch count equals PP world size.

**Solution:** Set `pipeline_parallel_size > 1` when `enable_pipefusion=True`.

### Issue: Import raises a mixin-order `TypeError`

**Cause:** `PipeFusionPipelineMixin` must be listed before `PipelineParallelMixin`.

**Solution:** Use:

```python
class YourPipeline(nn.Module, PipeFusionPipelineMixin, PipelineParallelMixin, CFGParallelMixin):
    ...
```

### Issue: Missing PipeFusion KV cache during patch mode

**Cause:** Async patch execution started before a warmup pass populated full K/V caches.

**Solution:** Use at least one warmup step. The runtime validates `pipefusion_warmup_steps >= 1`.

### Issue: PP communication hangs in patch mode

**Causes and checks:**

- Sender and receiver ranks are out of sync.
- A pending async send was not flushed before a later collective.
- CFG branch count changed unexpectedly.

---

## Reference Implementations

| Component                  | Path                                                                   | Notes                                                    |
|----------------------------|------------------------------------------------------------------------|----------------------------------------------------------|
| `PipeFusionPipelineMixin`  | `vllm_omni/diffusion/distributed/pipefusion/pipefusion.py`             | Pipeline wrapping, warmup split, async patch loop        |
| `PipeFusionRuntime`        | `vllm_omni/diffusion/distributed/pipefusion/pipefusion_runtime.py`     | Patch metadata, split dimension, cache key, buffer reset |
| `PipeFusionSchedulerMixin` | `vllm_omni/diffusion/distributed/pipefusion/pipefusion_scheduler.py`   | Per-patch scheduler caches and state gating              |
| Transformer mixins         | `vllm_omni/diffusion/distributed/pipefusion/pipefusion_transformer.py` | RoPE slicing, K/V cache patch updates, unpatchify        |
| Conv mixin                 | `vllm_omni/diffusion/distributed/pipefusion/pipefusion_conv.py`        | Conv3d activation cache and sliced convolution           |
| PP mixin                   | `vllm_omni/diffusion/distributed/pipeline_parallel.py`                 | `skip_sync`, `inter_comm_ids`, and `loopback_comm_id`    |
| PP tests                   | `tests/diffusion/distributed/test_pipeline_parallel.py`                | GPU coverage for labeled PP communication                |
| PipeFusion tests           | `tests/diffusion/distributed/test_pipefusion.py`                       | CPU coverage for PipeFusion bookkeeping                  |
| Wan2.2 T2V pipeline        | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`                 | Reference text-to-video integration                      |
| Wan2.2 I2V pipeline        | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py`             | Reference image-to-video integration                     |
| Wan2.2 transformer         | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py`              | Reference transformer integration                        |
