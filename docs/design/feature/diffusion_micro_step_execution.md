# Adding Micro-Step Execution Support for Diffusion Pipelines

Author-facing contract for implementing `stream_batch=True` (temporal
pipeline-parallel streaming diffusion) on a pipeline. Read
[Step Execution](diffusion_step_execution.md) and
[Pipeline Parallel](pipeline_parallel.md) first.

Reference implementation: `CausVidPipeline`
([`pipeline_causvid.py`](gh-file:vllm_omni/diffusion/models/causvid/pipeline_causvid.py)).

## Model

A pipeline supports micro-step execution iff it implements
`SupportsMicroStepExecution`
([`interface.py`](gh-file:vllm_omni/diffusion/models/interface.py)) and declares
`supports_micro_step_execution: ClassVar[bool] = True`. Unsupported pipelines
fail at load time, not on the first request.

Runtime limits: one in-flight request per engine (`max_num_seqs == 1`);
`cache_backend` is unsupported.

## The fixed ladder

A request of `num_chunks` chunks is denoised on a constant-shape ladder of
`ns = num_inference_steps` slots. Slot `j` holds a chunk at denoise level `j`;
each micro-step advances every slot one level and rolls the deepest (finished)
chunk off for decode.

The scheduler ([`stream_batch_scheduler.py`](gh-file:vllm_omni/diffusion/sched/stream_batch_scheduler.py))
emits one `RankTask` per PP rank per micro-step:

| Field | Meaning |
|-------|---------|
| `slot_chunks` | `list[int \| None]` of length `ns`; entry `j` is the chunk index at slot `j`, or `None` for an empty/dummy slot (PP fill/drain) |
| `is_last` | True on the final micro-step |

Chunk 0 is special: its micro-step is `slot_chunks = [0, None, …]` and the
runner routes it to `prepare_first_chunk`. The pipeline builds `slot_idxs` from
`slot_chunks` (`-1` for `None`); the model's rolling KV cache skips the write
for `slot_idx == -1` and the dummy row's output is discarded.

## Execution contract

`execute_micro_step`
([`diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py))
drives these methods:

- **New request:** `prepare_encode(state)` (one-time setup), then
  `set_pp_recv_dict_buffers(state)`.
- **Chunk 0** (`0 in slot_chunks`): `prepare_first_chunk(state)` — denoise
  chunk 0 alone through all `ns` steps on a clean KV cache, seed every slot
  from it (`state.seed_all_slots_from(0)`), then decode it (last rank).
- **Steady step** (every other micro-step):
  1. `prepare_chunks(state)` — rank 0 only: roll the `[ns, …]` buffer and admit
     the new chunk at slot 0.
  2. `denoise_step(state)` — all ranks: one transformer forward over the ladder.
  3. `step_scheduler(state, noise_pred)` — per-slot `scheduler.step`; advances
     `state.step_index`.
  4. `decode_chunks(state)` — last rank only: decode the deepest slot when it
     holds a real chunk; merge + return once all chunks are decoded.
  5. `prefetch_tensors(state, batch_size)` — pre-post the next-step recv.

## PP communication

Inherit `PipelineParallelMixin` + `CFGParallelMixin` and call the wrappers in
[`pipeline_parallel.py`](gh-file:vllm_omni/diffusion/distributed/pipeline_parallel.py),
never the coordinator primitives directly:

- `predict_noise_maybe_with_cfg(..., buf_idx, batch_size, preposted_its, use_buffer)`
- `scheduler_step_maybe_with_cfg(..., buf_idx, batch_size, receive_latents, use_buffer)`
- `prefetch_tensors_maybe_with_cfg(buf_idx, batch_size)`

Use `buf_idx = state.step_index % 2` consistently — recv buffers are
double-buffered so the in-flight recv doesn't clobber the slot being read.
`set_pp_recv_dict_buffers` pre-registers the `("latents", -1)` and
`("intermediate", 0)` schemas at the single fixed `batch_size = ns`; the first
send of an unseen schema would otherwise trigger a blocking exchange.

The forward chain (rank 0 → N-1) carries an `IntermediateTensors` holding
`hidden_states`, the per-slot timesteps `t`, and the latents `xt`. The last
rank exposes `t`/`xt` on the forward context (`ForwardContext.stream_t` /
`stream_xt`) so `step_scheduler` and the latent update read them without
re-deriving — see
[`forward_context.py`](gh-file:vllm_omni/diffusion/forward_context.py).

## Rules

- One forward per micro-step; only `step_scheduler` advances `state.step_index`.
- `prepare_first_chunk` uses `use_buffer=False` (non-buffered comm); steady
  steps use `use_buffer=True`..
- `None` slots are shape-only fillers (output discarded) — never write their
  KV (`slot_idx == -1`) and don't decode them.

## Validation

- `pipeline_parallel_size` of 1, 2, and >= 3 all complete.
- PP=1 and PP>1 outputs match (the `-1` sentinel + `stream_xt` flow is what
  keeps PP>1 non-first chunks correct).

## Related files

- Contract: [`interface.py`](gh-file:vllm_omni/diffusion/models/interface.py)
- Runner: [`diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Scheduler: [`stream_batch_scheduler.py`](gh-file:vllm_omni/diffusion/sched/stream_batch_scheduler.py)
- PP mixin: [`pipeline_parallel.py`](gh-file:vllm_omni/diffusion/distributed/pipeline_parallel.py)
- Reference: [`pipeline_causvid.py`](gh-file:vllm_omni/diffusion/models/causvid/pipeline_causvid.py)
