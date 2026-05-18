# Pipeline Parallel

This section describes how to add Pipeline Parallelism (PP) to a diffusion pipeline. We use the Wan2.2 text-to-video and
image-to-video pipelines as the reference implementations.

## Implementation Checklist

Adding Pipeline Parallel support requires:

1. ✅ **Inherit the mixin** - add `PipelineParallelMixin` before `CFGParallelMixin` in the pipeline class
2. ✅ **Make stage forward resumable** - support `intermediate_tensors` in the transformer forward path
3. ✅ **Return the right object type** - `IntermediateTensors` on non-last PP ranks, final model output on the last rank
4. ✅ **Use the standard denoising loop** - call `predict_noise_maybe_with_cfg()` and `scheduler_step_maybe_with_cfg()`
5. ✅ **Test parity** - compare PP results against the single-GPU baseline

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)

---

## Overview

### What is Pipeline Parallelism?

Pipeline Parallelism splits the denoising transformer into multiple sequential stages and places each stage on a
different rank. Instead of every rank holding the full DiT, each PP rank owns only a slice of the layers.

For each denoising step:

1. Rank 0 starts the forward pass using the current latents.
2. Each intermediate rank receives hidden states from the previous rank, runs its local layer slice, and forwards the
   intermediate tensors downstream.
3. The last PP rank produces the final noise prediction.
4. The last PP rank applies the scheduler step and sends the updated latents back to rank 0 for the next timestep.

This reduces per-rank model memory and enables larger diffusion transformers to run across multiple GPUs. It can also be
combined with CFG-Parallel, where each PP pipeline carries one CFG branch.

### Architecture

vLLM-Omni provides `PipelineParallelMixin` to encapsulate the PP communication pattern for diffusion pipelines.

| Method                            | Purpose                       | Automatic Behavior                                                              |
|-----------------------------------|-------------------------------|---------------------------------------------------------------------------------|
| `diffuse()`                       | Denoising loop boundary       | Wrapped by `PipelineParallelMixin` to flush pending async PP sends on exit      |
| `predict_noise_maybe_with_cfg()`  | Predict noise with PP support | Runs partial forwards on non-last PP ranks, combines with CFG logic when needed |
| `scheduler_step_maybe_with_cfg()` | Step scheduler with PP sync   | Runs scheduler on the last PP rank, returns updated latents to rank 0           |
| `_sync_pp_send()`                 | Flush pending async sends     | Waits for outstanding `isend` handles before later collectives or decode        |

`PipelineParallelMixin` is intentionally a pipeline-level abstraction. Your model-specific `predict_noise()` still
defines how a local stage executes.

### How It Works

`predict_noise_maybe_with_cfg()` automatically switches between these modes:

- **PP disabled** (`pipeline_parallel_size == 1`):
    - Falls back to `CFGParallelMixin.predict_noise_maybe_with_cfg()`
- **PP only** (`pipeline_parallel_size > 1`, `cfg_parallel_size == 1`):
    - Rank 0 starts with the input latents
    - Middle ranks receive `intermediate_tensors`, run their local layer range, and asynchronously send downstream
    - The last rank returns the final noise prediction
    - When CFG is enabled in this mode (sequential CFG), both the positive and negative branches run through the same
      PP pipeline, doubling the communication volume per denoising step. Prefer PP + CFG-Parallel when
      `cfg_parallel_size > 1` is available to avoid this overhead.
- **PP + CFG-Parallel** (`pipeline_parallel_size > 1`, `cfg_parallel_size > 1`):
    - Each PP pipeline carries one CFG branch
    - The last PP rank all-gathers across the CFG group
    - CFG combination happens on every last PP rank in the CFG group, matching the non-PP CFG-parallel behavior

`scheduler_step_maybe_with_cfg()` keeps the denoising loop consistent:

- **PP disabled**:
    - Falls back to `scheduler_step_maybe_with_cfg()`
- **PP enabled**:
    - Only the last PP rank has `noise_pred` and runs the scheduler step
    - The resulting latents are sent back to rank 0
    - Rank 0 receives an `AsyncLatents` wrapper that resolves only when the tensor is next consumed

This asynchronous design avoids unnecessary blocking between denoising steps.

When a pipeline class is defined, `PipelineParallelMixin` wraps its `diffuse()` method and calls `_sync_pp_send()` in a
`finally` block after `diffuse()` returns or raises. This keeps model code free of explicit PP cleanup while still
ensuring that the final non-blocking PP send completes before decode or later collectives.

---

## Step-by-Step Implementation

### Step 1: Inherit `PipelineParallelMixin` and `CFGParallelMixin`

`PipelineParallelMixin` **requires** `CFGParallelMixin` and must appear before it in the class MRO. This is enforced at
class definition time via `__init_subclass__`: defining a pipeline that inherits `PipelineParallelMixin` without
`CFGParallelMixin`, or after `CFGParallelMixin`, raises a `TypeError` immediately on import.

`PipelineParallelMixin` delegates noise prediction, CFG combination, and scheduler stepping to `CFGParallelMixin`, which
supplies `predict_noise()`, `predict_noise_maybe_with_cfg()`, `scheduler_step_maybe_with_cfg()`, and
`combine_cfg_noise()`.

**Example:**

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin
import torch.nn as nn


class YourPipeline(nn.Module, PipelineParallelMixin, CFGParallelMixin):
    ...
```

The order matters: `PipelineParallelMixin` must be listed before `CFGParallelMixin` so calls to
`predict_noise_maybe_with_cfg()` and `scheduler_step_maybe_with_cfg()` resolve to the PP-aware wrappers, while their
`super()` calls delegate to the CFG implementation when PP is disabled or after the last PP stage.

### Step 2: Make model forward and `predict_noise()` PP-aware

The PP mixin injects `intermediate_tensors` into the normal `predict_noise()` call. Your model forward path must support
two inputs:

- Normal input from rank 0, typically passed as `hidden_states` or `x`
- `intermediate_tensors` from upstream PP ranks

The standard model forward pattern is:

1. If `intermediate_tensors` is present, read the local hidden state from it.
2. Run only this rank's layer slice.
3. Return `IntermediateTensors(...)` on non-last PP ranks.
4. Return the final model output on the last PP rank. `CFGParallelMixin.predict_noise()` already follows this contract
   for common pipelines.

**Minimal example:**

```python
from vllm.sequence import IntermediateTensors
from vllm_omni.diffusion.distributed.parallel_state import get_pp_group


def forward(self, hidden_states=None, intermediate_tensors=None, **kwargs):
    if intermediate_tensors is not None:
        hidden_states = intermediate_tensors["hidden_states"]

    for i in range(self.start_layer, self.end_layer):
        hidden_states = self.layers[i](hidden_states)

    pp_group = get_pp_group()
    if not pp_group.is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states})
    return (hidden_states,)
```

### Step 3: Partition the transformer layers

The local module on each PP rank must expose only that rank's layer slice. In the reference tests and vLLM model
implementations, this is typically done with vLLM utilities such as `make_layers(...)`, which fill missing ranges with
`PPMissingLayer`. Besides partitioning the layers themselves, model authors should also wire
`make_empty_intermediate_tensors_factory(...)` for intermediate tensor allocation and `is_pp_missing_parameter(...)` for
PP-aware weight loading.

To prepare a transformer for PP, implement the following pieces in order.

#### 3.1 Split the transformer layers across PP ranks

Each PP rank should own only its local layer range, typically exposed as `[start_layer, end_layer)`. In practice this is
usually done with `make_layers(...)`, which constructs local layers and fills missing ranges with `PPMissingLayer`.

The goal is:

- every PP rank knows its `[start_layer, end_layer)` range
- non-local layers do not execute on this rank
- the forward can resume from incoming `intermediate_tensors`

By default, layers are distributed evenly across PP ranks using `get_pp_indices()`. When the number of layers is not
divisible by the PP size, remaining layers are assigned to middle partitions to balance compute and memory. You can
override this with the `VLLM_PP_LAYER_PARTITION` environment variable to specify an exact per-rank layer count:

```bash
# Example: 40 layers across 4 PP ranks, assigning 8 / 12 / 12 / 8 layers
export VLLM_PP_LAYER_PARTITION=8,12,12,8
```

The value must be a comma-separated list of integers whose length equals `pipeline_parallel_size` and whose sum equals
the total number of transformer layers.

#### 3.2 Expose `make_empty_intermediate_tensors`

Your transformer module should expose `self.make_empty_intermediate_tensors`, typically created with
`make_empty_intermediate_tensors_factory(...)`.

This is important because PP ranks need a consistent way to allocate placeholder `IntermediateTensors` with the expected
keys and hidden dimension.

**Example:**

```python
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory

self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
    ["hidden_states"],
    inner_dim,
)
```

For Wan2.2, the intermediate payload between PP stages is the token sequence stored under `"hidden_states"`, so the
factory is created with that key and the transformer hidden size.

#### 3.3 Return `IntermediateTensors` on non-last PP ranks

Your model `forward()` or custom `predict_noise()` implementation should consume `intermediate_tensors` on non-first
ranks and return `IntermediateTensors(...)` on non-last ranks.

This lets each PP stage resume from the upstream hidden states and pass the local result to the next stage.

#### 3.4 Skip non-local weights during `load_weights()`

When a model is PP-partitioned, many parameters in the checkpoint belong to layers that do not exist on the current
rank. `load_weights()` must therefore skip parameters for missing PP stages using `is_pp_missing_parameter(...)`.

If this is not done, weight loading will either fail or incorrectly try to load tensors into `PPMissingLayer`
placeholders.

The Wan2.2 transformer is the best reference here: it uses `is_pp_missing_parameter(...)` before loading both remapped
and fused parameters.

If your model has multiple transformer variants, PP still works as long as each selected transformer obeys the same
contract.

### Step 4: Use the standard denoising contract

vLLM-Omni diffusion pipelines already route denoising through `diffuse()`, `predict_noise_maybe_with_cfg()`, and
`scheduler_step_maybe_with_cfg()`. `PipelineParallelMixin` overrides those standard helpers when
`pipeline_parallel_size > 1`, so model integrations should not add separate PP-specific helper names or manual post-loop
synchronization.

In PP mode:

- Non-last PP ranks return `None` from `predict_noise_maybe_with_cfg()` after sending `IntermediateTensors` downstream.
- The last PP rank returns the final prediction, and with CFG-Parallel every last PP rank in the CFG group receives the
  combined result.
- Rank 0 receives `AsyncLatents` from `scheduler_step_maybe_with_cfg()`, which resolves only when the latents are
  consumed.
- Pending non-blocking sends are flushed automatically when `diffuse()` exits.

---

## Testing

Use an offline inference script with `pipeline_parallel_size > 1`.

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp2.mp4 \
--pipeline-parallel-size=2
```

For PP + CFG-Parallel together:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp2_cfg2.mp4 \
--pipeline-parallel-size=2 \
--cfg-parallel-size=2
```

**Verify:**

1. The run completes without hangs at the PP stage boundary
2. Output quality matches the non-PP baseline within normal numerical variance
3. Peak memory per GPU drops compared with the single-rank model
4. There are no outstanding communication errors before decode

---

## Troubleshooting

### Issue: `TypeError` on import — `CFGParallelMixin` missing

**Symptoms:** Importing a pipeline that inherits `PipelineParallelMixin` raises:

```
TypeError: YourPipeline inherits PipelineParallelMixin but not CFGParallelMixin.
```

or:

```
TypeError: YourPipeline must inherit PipelineParallelMixin before CFGParallelMixin ...
```

**Cause:** `PipelineParallelMixin` enforces via `__init_subclass__` that the subclass also inherits `CFGParallelMixin`
and lists `PipelineParallelMixin` first in the MRO.

**Solution:** Add `CFGParallelMixin` to the pipeline's base classes after `PipelineParallelMixin`:

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin


class YourPipeline(nn.Module, PipelineParallelMixin, CFGParallelMixin):
    ...
```

### Issue: Non-last PP ranks crash when calling `predict_noise`

**Symptoms:** Shape or missing-input errors on ranks other than the first or last PP rank.

**Cause:** The model forward path assumes direct input tensors and ignores `intermediate_tensors`.

**Solution:** Update the transformer `forward()` or custom `predict_noise()` path to load hidden states from
`intermediate_tensors` when present.

### Issue: PP outputs differ from single-GPU baseline

**Symptoms:** The PP run finishes but produces numerically inconsistent outputs.

**Causes & Solutions:**

- **Local layer partitioning is wrong**
    - Verify each rank runs only its `[start_layer, end_layer)` slice
- **Non-last ranks return a plain tensor instead of `IntermediateTensors`**
    - Return `IntermediateTensors({...})` until the last PP stage
- **CFG branch wiring is incorrect**
    - When CFG is enabled, confirm positive and negative kwargs are passed exactly as in the non-PP path

---

## Reference Implementations

Complete examples in the codebase:

| Component               | Path                                                       | Notes                                                                                                                   |
|-------------------------|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `PipelineParallelMixin` | `vllm_omni/diffusion/distributed/pipeline_parallel.py`     | Core PP communication and scheduler helpers                                                                             |
| `CFGParallelMixin`      | `vllm_omni/diffusion/distributed/cfg_parallel.py`          | Default `predict_noise()` tuple normalization and CFG helper fallback                                                   |
| Wan2.2 transformer      | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py`  | Reference for layer partitioning, `IntermediateTensors`, `make_empty_intermediate_tensors`, and PP-aware weight loading |
| Wan2.2 T2V pipeline     | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`     | Reference PP + CFG integration for text-to-video                                                                        |
| Wan2.2 I2V pipeline     | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py` | Reference PP + CFG integration for image-to-video                                                                       |
| PP tests                | `tests/diffusion/distributed/test_pipeline_parallel.py`    | Baseline parity and async communication tests                                                                           |
