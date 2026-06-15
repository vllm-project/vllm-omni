# DiT Cache Pool for Step-Wise Batching

This document describes the DiT Cache Pool extension built on top of
[Diffusion Step Execution](diffusion_step_execution.md) and
[Continuous Batching for Step-Wise Diffusion](diffusion_continuous_batching.md).
The base step-execution contract is unchanged. DiT Cache Pool adds
request-local Cache-DiT state management so compatible requests can share one
step-wise denoise forward while keeping independent cache decisions.

## Why It Helps

Continuous batching lets the scheduler group compatible diffusion requests into
one `denoise_step(input_batch)` call. Cache-DiT acceleration also needs mutable
state across denoise steps: cached block outputs, residual history, warmup
state, and hit/miss decisions.

Without request-local cache slots, those mutable DiT cache objects would be
shared globally on the pipeline and could bleed across requests. DiT Cache Pool
keeps the acceleration state attached to each live request and installs the
right state before each scheduler tick.

This enables:

- grouped Cache-DiT denoise steps for homogeneous request batches
- independent all-miss, all-hit, and partial-hit behavior inside one batch
- safe request join/finish behavior when batch rows change over time
- cleanup of resident cache state when a request finishes or is interrupted

## Overview

Each live
[`DiffusionRequestState`](gh-file:vllm_omni/diffusion/worker/utils.py)
owns an optional `cache_slot`. The slot is a
[`CacheBackendSlot`](gh-file:vllm_omni/diffusion/worker/utils.py) containing
backend-owned payload, compatibility metadata, and a lightweight resident-byte
estimate.

The runner does not inspect backend internals. It delegates cache lifecycle
to [`DiTCacheManager`](gh-file:vllm_omni/diffusion/cache/dit_cache_manager.py),
which uses a backend-specific `DiTCacheStateDriverBase` implementation:

- Cache-DiT uses
  [`CacheDiTStateDriver`](gh-file:vllm_omni/diffusion/cache/cache_dit_driver.py).
- TeaCache uses
  [`TeaCacheStateDriver`](gh-file:vllm_omni/diffusion/cache/teacache/driver.py)
  for the single-request step-wise slot path.

For Cache-DiT grouped batching, the manager activates all scheduled request
slots before the runner calls the batched `denoise_step(input_batch)`. The
Cache-DiT driver then maps each input row to the correct per-request cache
context.

The normal runner execution shape remains one batched denoise call per
scheduler tick. DiT Cache Pool only adds cache activation and cleanup around
that call.

## Enablement

Use the existing diffusion engine arguments. No extra public flag is added:

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

omni = AsyncOmni(
    model="Qwen/Qwen-Image",
    step_execution=True,
    cache_backend="cache_dit",
    max_num_seqs=4,
    diffusion_batch_size=4,
)
```

For serving, use the existing step-execution and Cache-DiT flags, then raise
`--max-num-seqs` above `1`:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --cache-backend cache_dit \
  --max-num-seqs 4
```

`max_num_seqs=1` keeps the single-request step-wise path. Grouped Cache-DiT
batching requires both step execution and a scheduler capacity greater than
one.

## Slot Lifecycle

The slot lifecycle is request-local:

1. A new request enters the runner state cache.
2. `DiTCacheManager.activate(states)` creates or restores each request slot.
3. Fresh slots are initialized with the request's `num_inference_steps`.
4. The Cache-DiT driver installs per-request cache contexts for the active
   batch.
5. The runner executes one `denoise_step(input_batch)`.
6. `DiTCacheManager.deactivate(states)` captures mutable cache state back into
   the request slots.
7. Completed, aborted, or interrupted requests free their slots.

Compatibility is checked by the driver. If an existing slot does not match the
current request shape or step count, the manager clears it and creates a fresh
slot for that request.

## Batch Context

Cache-DiT grouped batching uses driver-level batch activation:

- `install_batch_slots(states)` installs all request contexts for one forward.
- `deactivate_batch_slots()` captures and clears the active batch context.

The Cache-DiT driver builds row-aligned context metadata from the scheduled
states. This is what lets one batched forward contain different cache outcomes
per request: one row can be a cache hit while another row in the same batch is
a miss.

Backends without batched activation keep the single-request step-wise
slot-switching path. They do not silently turn one multi-request scheduler
batch into serial single-request forwards.

## Runner Integration

The runner-side integration is intentionally small. In
[`DiffusionModelRunner.execute_stepwise()`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py),
the scheduled states are updated and packed into one
[`InputBatch`](gh-file:vllm_omni/diffusion/worker/input_batch.py). The helper
`_denoise_step_with_cache()` then wraps the normal denoise forward:

1. Build attention metadata for the current `InputBatch`.
2. Activate cache slots if a cache backend is enabled.
3. Run `pipeline.denoise_step(input_batch)`.
4. Deactivate cache slots in a `finally` block.

After the denoise forward, the existing step-wise path slices `noise_pred`
back to request rows, advances each request scheduler, decodes completed
requests, and frees slots for requests that leave the runner state cache.

This keeps cache logic separate from batch scheduling, request admission, and
pipeline denoise semantics.

## Coverage

The current tests cover:

- per-request cache slot initialize, activate, deactivate, and free behavior
- Cache-DiT batched forward for all-miss, all-hit, and partial-hit batches
- independent cache decisions inside one scheduler step
- batch-aligned kwargs slicing and encoder-cache trimming
- dynamic row layout when requests join or finish at different step positions
- single-request step-wise behavior for cache backends without batched
  activation
- grouped Cache-DiT E2E coverage with `batch_size=4`

## Current Limitations

- Only homogeneous batches admitted by the step-wise diffusion scheduler are
  supported.
- Multi-request Cache-DiT batching requires backend support for batched slot
  activation.
- TeaCache keeps the single-request step-wise slot path and does not support
  grouped batch activation yet.
- Request-mode diffusion and AR/KV cache management are separate systems.
- The design is scoped to native step-wise diffusion pipelines; unsupported
  pipelines still fail early during model loading.

## Related Files

- Runner:
  [`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Request state and cache slot:
  [`vllm_omni/diffusion/worker/utils.py`](gh-file:vllm_omni/diffusion/worker/utils.py)
- Input batch:
  [`vllm_omni/diffusion/worker/input_batch.py`](gh-file:vllm_omni/diffusion/worker/input_batch.py)
- DiT cache manager:
  [`vllm_omni/diffusion/cache/dit_cache_manager.py`](gh-file:vllm_omni/diffusion/cache/dit_cache_manager.py)
- Cache-DiT driver:
  [`vllm_omni/diffusion/cache/cache_dit_driver.py`](gh-file:vllm_omni/diffusion/cache/cache_dit_driver.py)
- TeaCache driver:
  [`vllm_omni/diffusion/cache/teacache/driver.py`](gh-file:vllm_omni/diffusion/cache/teacache/driver.py)
- Tests:
  [`tests/diffusion/test_dit_cache_pool.py`](gh-file:tests/diffusion/test_dit_cache_pool.py)
