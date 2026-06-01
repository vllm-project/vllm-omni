# Adding Micro-Step Execution Support for Diffusion Pipelines

This guide documents vLLM-Omni's micro-step diffusion contract for model
authors and contributors implementing `stream_batch=True` support for a
diffusion pipeline.

For end-user enablement, supported models, and current limitations, see
[Micro-Step Execution](../../user_guide/diffusion/micro_step_execution.md).

This document describes the micro-step execution contract only. It builds on
the request-/step-level contract in
[Step Execution](diffusion_step_execution.md) and the PP partitioning rules in
[Pipeline Parallel](pipeline_parallel.md). Read those first.

## Current Support Scope

`stream_batch` is **not** a generic diffusion toggle. It only works for
pipelines that implement the segmented stateful contract in
[`vllm_omni/diffusion/models/interface.py`](gh-file:vllm_omni/diffusion/models/interface.py)
as `SupportsMicroStepExecution`.

This page is intentionally author-facing. Treat runtime enablement
(`stream_batch=True` when constructing `Omni`) as an opt-in user knob layered
on top of the implementation contract below.

Current in-tree support:

| Pipeline | Example models | Micro-step execution |
|----------|----------------|----------------------|
| `LingbotWorldFastPipeline` | `lingbot_world/lingbot-world-base-cam/Lingbot-World-Fast` | Yes |
| All other diffusion pipelines | — | No |

Current engine/runtime limitations:

- `max_num_seqs == 1` — exactly one in-flight request per engine.
- `cache_backend` is not supported.
- Unsupported pipelines fail early during model loading instead of
  failing on the first request.

## Execution Contract

Micro-step mode is driven by seven pipeline methods plus the shared mutable
request state object:

- `prepare_encode(state)`: one-time request preparation (inherited from
  step execution).
- `set_pp_recv_dict_buffers(state)`: register PP recv buffers and schema
  cache for every `(name, segment_idx, batch_size)` this request will use.
- `encode_chunk_inputs(state, new_idxs)`: per-chunk latent initialization.
  Returns a tensor stacked along dim 0 over `new_idxs`; the runner stitches
  it onto `state.latents` and into each chunk's `chunk.latents`.
- `denoise_step(state, batch_size)`: row-batched noise prediction over
  `batch_size` chunks at different denoising step indices.
- `step_scheduler(state, noise_pred, per_request_scheduler, batch_size)`:
  per-row scheduler update on the last rank; sends the updated latents
  back to rank 0 via the ring (rank 0 picks them up via `prefetch_tensors`,
  not inside this call). Every rank increments `state.step_index`.
- `prefetch_tensors(state, batch_size)`: pre-post the next-step recv on the
  comms stream so it overlaps with this rank's compute.
- `post_decode(state)`: incremental decode of one or more freshly-finished
  chunks (called whenever the previous tick produced `finished_idxs`).

The state lives in
[`vllm_omni/diffusion/worker/utils.py`](gh-file:vllm_omni/diffusion/worker/utils.py)
as `DiffusionRequestState` plus per-chunk `ChunkState` entries under
`state.extra["chunks"]`.

The worker-side micro-step loop lives in
[`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
under `execute_micro_step`:

1. `prepare_encode()` runs once for a new request.
2. `set_pp_recv_dict_buffers()` runs immediately after, before any P2P.
3. Each micro-step:
   - Rank 0 calls `post_decode()` for any chunks the previous tick
     reported as finished, and accumulates the decoded output.
   - Rank 0 and rank N-1 call `encode_chunk_inputs()` for their layout's
     `new_idxs`. On rank 0 those are chunks freshly admitted this tick;
     on rank N-1 they are the same chunks arriving at the back of the
     ring N-1 ticks later — both ranks must produce identical initial
     noise so the scheduler step on the last rank starts from the same
     latents the first rank started from.
   - All ranks with `chunk_indices` non-empty call `denoise_step()` then
     `step_scheduler()`. The last rank also snapshots
     `chunk.latents = state.latents[i:i+1]` per row so the next time those
     chunks reach the last rank they can resume.
   - `prefetch_tensors()` runs sized to the previous rank's load so the
     next recv is posted before the next micro-step's compute.

## Per-Rank Chunk Layout

`StreamBatchScheduler` builds one `RankTask` per PP rank per micro-step:

| Field | Meaning |
|-------|---------|
| `chunk_indices` | Chunks this rank will denoise this tick |
| `layout.circulating_idxs` | Chunks that drained from rank N-1 last tick still needing more steps, looping back to rank 0 |
| `layout.finished_idxs` | Chunks that completed `num_inference_steps` at rank N-1 last tick, ready for decode |
| `layout.new_idxs` | Chunks freshly admitted at rank 0 (up to SLO `B_target`, capped by `num_chunks - admitted_so_far`) |

Layouts travel with their chunks: at rank R the current layout was built at
rank 0 R ticks ago, so `new_idxs` at rank R names the chunks admitted R ticks
ago and now reaching this rank for the first time on their first lap.

The runner uses rank 0's layout to assemble `state.latents` along dim 0 from
the circulating snapshot + fresh-noise rows for `new_idxs`, and to
incrementally decode `finished_idxs`. The last rank does the same assembly
when it owns `new_idxs` so step_scheduler has the matching initial latents.


## Recommended Split

| Request-level phase | Micro-step method | What belongs there |
|---------------------|-------------------|--------------------|
| Input validation, prompt encoding, timestep prep, per-request scheduler | `prepare_encode()` | Anything that should happen once per request |
| PP recv buffer / schema registration for every `(name, segment_idx, B)` | `set_pp_recv_dict_buffers()` | Iterate `1..slo_max_batch * num_inference_steps` |
| Per-chunk latent init (fresh randn, V2V VAE encode, anchor latents, plucker, etc.) | `encode_chunk_inputs()` | Build per-chunk initial latents (RNG must match across rank 0 and rank N-1); write per-chunk conditioning into `state.extra["chunks"][idx].extra` only on the rank that will read it |
| Row-batched transformer forward | `denoise_step()` | Row-aware kwargs, `predict_noise_maybe_with_cfg(buf_idx=step_index % 2, batch_size=B, preposted_its=...)` |
| Per-row `scheduler.step` and `state.step_index += 1` | `step_scheduler()` | `scheduler_step_maybe_with_cfg(..., receive_latents=False, batch_size=B)` |
| Pre-post next-step recv | `prefetch_tensors()` | `prefetch_tensors_maybe_with_cfg(buf_idx=step_index % 2, batch_size=B)` and stash on state |
| Per-chunk VAE decode | `post_decode()` | Decode the leading `len(finished_idxs)` rows of `state.latents` (runner narrows the slice for you) |

Keep the micro-step path reusing the same helpers as the request-level path
whenever possible. Reimplementing the denoise loop from scratch is the easiest
way to introduce behavioral drift.

## PP Communication

`PipelineGroupCoordinator` provides three primitives the micro-step path
leans on:

| Primitive | Purpose |
|-----------|---------|
| `set_recv_dict_buffer(name, segment_idx, template_dict, batch_size)` | Register the schema and pre-allocate a double-buffer pair (slots 0 and 1) for one logical channel |
| `pipeline_isend_tensor_dict(...)` | Async send of an arbitrary dict to the next rank |
| `pipeline_irecv_tensor_dict(..., buf_idx)` | Posts async recv into the pre-allocated buffer slot; returns an `AsyncIntermediateTensors`/`AsyncLatents` that defers `.wait()` until consumed |

[`PipelineParallelMixin`](gh-file:vllm_omni/diffusion/distributed/pipeline_parallel.py)
already wraps these in `predict_noise_maybe_with_cfg`,
`scheduler_step_maybe_with_cfg`, and `prefetch_tensors_maybe_with_cfg`.
Pipelines should call those, not the coordinator primitives directly.

### Why schemas must be pre-registered

The first call to `pipeline_isend_tensor_dict` on a previously unseen
`(name, segment_idx, batch_size)` triggers a blocking schema exchange.
`set_pp_recv_dict_buffers` populates the cache identically on all ranks so the
schema path is never entered during the data loop.

Enumerate every `B` the request can hit. For SLO-driven admission the upper
bound is `slo_max_batch * num_inference_steps`.

### Double buffering

Caller picks `buf_idx = state.step_index % 2` consistently across
`denoise_step`, `step_scheduler`, and `prefetch_tensors` on the same
micro-step. Alternating slots keeps the previous result readable while the
next recv is in flight.

## Row-Batched Computation

`state.batched_timesteps` is a 1-D tensor of length `B`; row `i` carries
`state.timesteps[chunks[i].step_index]`. Inside `denoise_step` and
`step_scheduler`, treat the leading dim as a mix of independent chunks at
*different* progress points.

## Lingbot Reference

[`pipeline_lingbot_world_fast.py`](gh-file:vllm_omni/diffusion/models/lingbot_world_fast/pipeline_lingbot_world_fast.py)
is the reference for the *self-forcing* pattern and is split
correctly for the current contract:

- `prepare_encode()` wraps `self.scheduler` in `LingbotFlowScheduler` so the
  last denoise step returns the cached x0 and intermediate steps re-noise to
  the next `t`. Two `torch.Generator`s are created on every rank: `seed_g`
  for chunk noise (consumed identically on every rank that calls
  `encode_chunk_inputs`) and `seed_g_addnoise` for the re-noise step
  (consumed only on the last rank).
- `set_pp_recv_dict_buffers()` registers `("latents", -1, B)` and
  `("intermediate", 0, B)` templates for every B in
  `1..slo_max_batch * num_inference_steps`.
- `encode_chunk_inputs()` builds per-chunk noise on every rank using
  `seed_g`. Only rank 0 (first stage) additionally stream-encodes per-chunk
  `y` (with anchor-frame handling on the first chunk) and computes Plucker
  embeddings, stashing both into `state.extra["chunks"][idx].extra` for
  `denoise_step` to read.
- `denoise_step()` slices per-row `current_starts`, `y`, and
  `c2ws_plucker_emb` from `state.extra["chunks"][idx]` keyed by the current
  micro-step's `chunk_idxs`, then calls
  `predict_noise_maybe_with_cfg(...)`. The per-chunk conditioning is only
  read on the first stage; the last stage receives processed hidden states
  via intermediate tensors.
- `step_scheduler()` rides the shared `scheduler_step_maybe_with_cfg(...,
  receive_latents=False, batch_size=B, generator=state.extra["seed_g_addnoise"])`
  and bumps `state.step_index`.
- `prefetch_tensors()` calls
  `prefetch_tensors_maybe_with_cfg(buf_idx=state.step_index % 2,
  batch_size=B)` and stashes results into `state.latents` (rank 0) or
  `state.extra["preposted_its"]` (others).

That decomposition is the target pattern for future micro-step models.

## Rules For New Pipelines

- Inherit `PipelineParallelMixin` and `CFGParallelMixin`.
- Declare `supports_micro_step_execution: ClassVar[bool] = True` on the
  pipeline class.
- Pre-populate every `(name, segment_idx, batch_size)` in
  `set_pp_recv_dict_buffers`. Skipping a `B` triggers the blocking schema
  path and risks PP deadlock.
- Use `state.extra["chunks"][idx]` (a `ChunkState`) for per-chunk persistent
  state: latents snapshot at the last rank, per-chunk scheduler, conditioning
  slices.
- Do not put request-scoped scheduler state on `self.scheduler`. Deep-copy
  it into `state.scheduler` during `prepare_encode` (the runner then
  deep-copies that into each new `ChunkState.scheduler` on admission).
- Do not mutate `state.step_index` inside `denoise_step`. Only
  `step_scheduler` should advance it.
- Use `buf_idx = state.step_index % 2` across `denoise_step`,
  `step_scheduler`, and `prefetch_tensors`.

## Validation Checklist

Before marking a pipeline `supports_micro_step_execution = True`, verify:

- `pipeline_parallel_size=2` and `pipeline_parallel_size>=3` both complete.
- `B=1` and `B>1` outputs match — verifies per-row scheduler / cache /
  conditioning slicing.
- CFG-parallel and non-CFG paths both work if the pipeline supports them.

## Related Files

- Contract: [`vllm_omni/diffusion/models/interface.py`](gh-file:vllm_omni/diffusion/models/interface.py)
- State: [`vllm_omni/diffusion/worker/utils.py`](gh-file:vllm_omni/diffusion/worker/utils.py)
- Runner loop: [`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Scheduler: [`vllm_omni/diffusion/sched/stream_batch_scheduler.py`](gh-file:vllm_omni/diffusion/sched/stream_batch_scheduler.py)
- PP coordinator: [`vllm_omni/diffusion/distributed/group_coordinator.py`](gh-file:vllm_omni/diffusion/distributed/group_coordinator.py)
- PP mixin: [`vllm_omni/diffusion/distributed/pipeline_parallel.py`](gh-file:vllm_omni/diffusion/distributed/pipeline_parallel.py)
- Reference pipeline: [`vllm_omni/diffusion/models/lingbot_world_fast/pipeline_lingbot_world_fast.py`](gh-file:vllm_omni/diffusion/models/lingbot_world_fast/pipeline_lingbot_world_fast.py)
