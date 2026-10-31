# Diffusion Continuous Batching

This document describes the unified diffusion execution architecture, including
request execution, request-level batching, step execution, continuous batching,
and output delivery. For user-facing configuration and CLI examples, see
[Diffusion Execution Modes](../../user_guide/diffusion/execution_modes.md).

## Overview

`DiffusionEngine` selects an execution mode from `step_execution` and uses
`max_num_seqs` as that mode's scheduler capacity:

| Configuration | Engine mode | Scheduler | Execution |
|---|---|---|---|
| `step_execution=False`, `max_num_seqs=1` | `REQUEST_BATCH` | `RequestScheduler` | One complete request-level `forward()` |
| `step_execution=False`, `max_num_seqs>1` | `REQUEST_BATCH` | `RequestScheduler` | One fused `forward()` over compatible requests |
| `step_execution=True`, `max_num_seqs=1` | `STEP_BATCH` | `StepScheduler` | One request advanced one denoise step per scheduler tick |
| `step_execution=True`, `max_num_seqs>1` | `STEP_BATCH` | `StepScheduler` | Compatible requests advanced together in step waves |

Serial request execution is handled inside `REQUEST_BATCH`; it is not a
separate engine mode. Similarly, continuous batching is the multi-request
configuration of `STEP_BATCH`, not an independent execution mode.

The engine performs the following initialization:

1. Resolve model-specific pre- and post-processing hooks.
2. Select `REQUEST_BATCH` or `STEP_BATCH`.
3. Construct the configured `DiffusionExecutor`.
4. Construct `RequestScheduler` or `StepScheduler`, unless a scheduler was
   explicitly injected.
5. Bind `execute_batch` or `execute_step`.

`DiffusionEngine.make_engine()` owns startup warmup after engine construction.
This keeps construction, backend selection, and warmup failure cleanup in one
lifecycle.

## Unified Output Stream

All asynchronous diffusion requests use the same per-request output stream,
regardless of whether the caller exposes streaming results to the user.

```python
async for output in engine.step_streaming(request):
    ...
```

When a request is admitted, the engine creates an
`asyncio.Queue[DiffusionOutput]` keyed by its request ID. The busy loop sends
both intermediate chunks and final results through that queue:

```text
request
  -> scheduler.add_request()
  -> scheduler.schedule()
  -> execute_batch() or execute_step()
  -> scheduler.update_from_output()
  -> per-request output queue
  -> step_streaming()
```

Streaming callers forward every yielded output. Non-streaming callers consume
the same generator to completion and return only its final output. This avoids
maintaining separate future-based and queue-based result paths.

`DiffusionEngine.step()` and
`async_add_req_and_wait_for_response()` remain deprecated compatibility
wrappers. New integrations should consume `step_streaming()` or
`async_add_req_and_stream_response()`.

### Completion and Cancellation

Output delivery and scheduler lifecycle are related but separate:

- A terminal scheduler result always finalizes request state.
- A disconnected or cancelled consumer removes only its delivery queue.
- The scheduler may still finish and clean up the request after its queue has
  been removed.
- Engine shutdown sends an error output to all remaining streams.

This separation prevents cancelled consumers from leaving active request IDs in
the scheduler.

## Request-Batch Execution

Request-batch execution runs the complete diffusion pipeline for each scheduler
wave. Each `OmniDiffusionRequest` remains an independent logical request with
its own prompt, sampling parameters, seed, request ID, output, error, and abort
state.

With `max_num_seqs=1`, `execute_batch()` handles the serial request path. With a
larger value, a pipeline can opt into fused execution by declaring:

```python
supports_request_batch = True
```

Its `forward()` method must accept `DiffusionRequestBatch` and return one
`DiffusionOutput` per request. The runner validates the result count and maps
outputs back to their original request IDs with `BatchRunnerOutput`.

Current in-tree request-batch implementations include the Flux, LTX-2, SD3, and
Qwen-Image pipelines. Treat the pipeline capability flag in the source as
authoritative because support continues to expand.

### Request-Batch Data Flow

- `DiffusionSchedulerOutput` carries scheduled request IDs and payloads.
- `DiffusionRequestBatch` provides the pipeline-facing static batch.
- `BatchRunnerOutput` carries per-request results back to the engine.

Request-local setup, such as seed handling and output/error mapping, remains
per request. Cache refresh, homogeneous LoRA activation, and
`pipeline.forward(req_batch)` can be shared by the batch.

### Compatibility and Admission

`RequestBatchSamplingParamsKey` contains fields that affect the tensor contract,
including shape, guidance, output count, and LoRA identity. Only compatible
requests can share a fused forward call.

Admission is FIFO and conservative. An incompatible request at the front of the
waiting queue can prevent later compatible requests from joining the current
wave.

`request_batch_max_wait_ms` optionally delays the first schedule operation of a
new wave so bursty arrivals can accumulate. It applies only when:

- the pipeline supports request batching;
- step execution is disabled;
- no requests are currently running; and
- the configured wait is greater than zero.

The wait ends when the queue reaches capacity, stabilizes briefly, reaches its
deadline, or the engine stops.

## Step-Batch Execution

Step execution exposes denoising progress to the scheduler. A supporting
pipeline implements four stateful operations:

| Operation | Responsibility |
|---|---|
| `prepare_encode(state)` | Validate input, encode prompts, initialize latents and timesteps, and create request-local scheduler state |
| `denoise_step(input_batch, *, states=...)` | Run one denoise forward for the scheduler-provided request states |
| `step_scheduler(state, noise_pred)` | Update latents and advance request progress |
| `post_decode(state)` | Decode and postprocess a completed request or output boundary |

Persistent request state lives in `StepRequestState`. Pipeline-specific fields
that do not belong in the shared contract should be stored in `state.extra`.
Queueing and lifecycle metadata remain in the scheduler's request state.

Current native pipelines that explicitly enable step execution include
Qwen-Image, BAGEL, HunyuanImage3, and Helios. Step execution alone does not
imply continuous-batching support: Qwen-Image accepts batched step states.
BAGEL accepts batched step states for image generation with its default,
think-mode, and single-stage deploy configs. Only the diffusion stage uses the
step protocol in two-stage deployments; explicit single-stage text output
retains complete-request execution. BAGEL requires at least two inference steps
because it schedules `num_inference_steps - 1` denoising updates, and its step
path does not currently support sequence parallelism or a diffusion cache
backend. When a finished stepwise request is followed by an explicit text
fallback wave, the runner retires its step state and paged diffusion-KV rows
before the full forward dispatch.
HunyuanImage3 accepts batched step states only when its resolved self-attention
backend is `TORCH_SDPA`; otherwise it rejects groups larger than one request.
Configure `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` or
`diffusion_attention_config.default.backend=TORCH_SDPA` when
`max_num_seqs>1`. See the
[HunyuanImage-3.0 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md)
for its validated configuration. Helios supports only a single active step
request and must use `max_num_seqs=1`.

### Continuous Batching

When `max_num_seqs>1`, `StepScheduler` can keep multiple compatible requests
active. The runner gathers their state into `InputBatch`, performs one batched
denoise forward, then applies scheduler updates per request:

1. Receive transferred KV payloads, then run `prepare_encode()` for newly
   admitted requests.
2. Build or refresh `InputBatch`.
3. Run one batched `denoise_step(input_batch, states=states)`.
4. Slice noise predictions back to each request.
5. Run each request's `step_scheduler()`.
6. Run `post_decode()` at a chunk boundary or request completion.
7. Scatter updated latents back into persistent request state.

`StepBatchSamplingParamsKey` protects the batched tensor contract. Requests can
have different total step counts and current step indices, but shape-sensitive,
CFG-sensitive, and LoRA fields must be compatible. FIFO head-of-line blocking
also applies here.

Chunk-capable pipelines may emit intermediate results. Final-only step
pipelines emit when the request finishes even if `streaming_output=True`.

### MixFusion (Mixed-Resolution Step Batching)

`StepBatchSamplingParamsKey` includes height/width/resolution, so plain step
batching never co-schedules mixed resolutions. MixFusion relaxes those three
fields out of the key so different resolutions can share one step batch, at the
cost of a pipeline contract the scheduler must not assume generically:

- A pipeline opts in with `supports_mixfusion: ClassVar[bool] = True`
  (read via `getattr`, not part of the runtime protocol check). Currently
  only `QwenImagePipeline` sets it.
- The engine resolves the capability through the registry at startup and
  rejects server-level `enable_mixfusion` on pipelines without it. Because
  resolution only leaves the step-batch key under the `StepScheduler`,
  server-level `enable_mixfusion` also enables step execution automatically
  (same coupling as `streaming_output`).
- The scheduler resolves the capability in `initialize()` and only then
  applies the shape relaxation. On an unsupported pipeline, a request-level
  `extra_args={"enable_mixfusion": True}` degrades to a no-op: resolution
  stays in the key and per-prompt `additional_information` requests keep
  their key.

With the capability present, the model runner hands every denoise step to
`pipeline.mixfusion_driver`, a model-agnostic `MixFusionDriver` that
`registry._apply_mixfusion_if_enabled` mounts at model load. The driver
dispatches on the batch: dense (tensor) latents go straight to
`pipeline.denoise_step`; list latents return `None` when the pipeline is
interrupted and otherwise take one packed forward only when
`MixFusionDriver.admit` accepts them. Rejected batches fall back to `serial_ragged`, which rebuilds one
single-request dense batch per request and calls the same `denoise_step`
with flattened latents, so per-image conditioning stays aligned.

Admission rejects, in order:

- a transformer without a `mixfusion_adapter`;
- a transformer whose `_mixfusion_plan` hooks were not applied
  (`_mixfusion_hooks_applied`);
- a non-empty `pipeline.attention_kwargs` (the packed forward has no
  `attention_kwargs` channel, so the serial path threads them into every
  block instead);
- `sequence_parallel_size > 1` (packing already splits and stacks the image
  stream; SP would shard rows a second time);
- `do_true_cfg` and image latents (edit path);
- foreign forward hooks on any module the plan touches, reported by
  `plan_foreign_hooks` as `<plan name>:torch-native` (layerwise offload,
  FSDP2 or cache hooks would see a layout they were not written for);
- a chunk policy the layout cannot meet: the greatest common divisor of the
  per-request sequence lengths is the chunk size, which must be at least
  `mixfusion_min_chunk_tokens`, and the total chunk count must not exceed
  `mixfusion_max_chunks`. Per-request extras are aggregated conservatively
  (max of the minimums, min of the maximums; defaults 256 and 128), so no
  request's limits are violated by fusing the batch.

Backends without packed varlen support (e.g. NPU)
are served by the attention layer's automatic fallback: documents are grouped
by length into mask-free dense forwards, so the packed dispatch stays active
and documents remain isolated without a per-request loop.

Known limitation: the packed path does not thread `attention_kwargs`
into the packed forward. To keep them from being dropped silently, the
driver falls back to the serial ragged path when they are set; features
that rely on them (e.g. LoRA scales, TeaCache) therefore lose MixFusion
batching until the packed call grows an `attention_kwargs` channel.

#### Adapter / driver layering

MixFusion is split into three layers so that adding a model touches only the
model-specific one:

| Layer | Location | Owns |
|---|---|---|
| Driver | `vllm_omni/diffusion/mixfusion/driver.py` (`MixFusionDriver`) | Step dispatch, admission, chunk-policy aggregation, arming/restoring `MixFusionStep`, serial-ragged fallback |
| Hooks + execution plan | `vllm_omni/diffusion/hooks/mixfusion.py`, `mixfusion/plan.py`, `mixfusion/execution_plan.py` | Applying `_mixfusion_plan` at module boundaries, the chunk-stacked layout (`PackedExecutionPlan`, gather/scatter, chunk split/reassembly), carrying opaque `PackedInputs` payloads |
| Adapter | `vllm_omni/diffusion/models/<model>/mixfusion_adapter.py` | The model's forward signature and packed-call shape, attention-stream declaration, packing/preparation and output restoration |

Framework responsibilities:

- `registry._apply_mixfusion_if_enabled` mounts the driver and calls
  `apply_mixfusion(transformer, plan)`; the hooks stay inert unless the
  driver arms a `MixFusionStep`.
- The driver never interprets the model-specific payloads it carries
  (`PackedInputs.boundary_outputs` / `adapter_state`); it only moves them
  between the hooks the plan declares.
- The layout primitives (`gcd_chunk_size`, `plan_chunk_size`,
  `build_execution_plan`, `split_sequence_chunks`, `expand_per_chunk`,
  `reassemble_chunk_rows`) know only `AttentionLayout`/`StreamSpec` and
  tensors.

Model responsibilities:

- Implement the `MixFusionAdapter` protocol: `build_packed_call`,
  `can_pack`, `build_attention_layout`, `pack_inputs`,
  `prepare_position_embeddings`, `restore_output`. The protocol is
  structural (`runtime_checkable`): an adapter only needs the methods, not
  a base class.
- Declare `_mixfusion_plan` on the transformer and register the adapter as
  `mixfusion_adapter`.
- Prepare the step-invariant position embeddings in
  `prepare_position_embeddings` and call it from `pack_inputs`; the
  framework never interprets that payload.
- Keep mixed-resolution attention in the model's attention leaf, which reads
  the stream layout from the armed step instead of a forward argument.

`QwenImageMixFusionAdapter` is the reference implementation. Porting a model
therefore means adding `models/<model>/mixfusion_adapter.py`, registering the
adapter and `_mixfusion_plan` in `transformer.py`, adding a model test under
`tests/diffusion/models/<model>/`, and declaring
`supports_mixfusion: ClassVar[bool] = True` on the pipeline (the engine and
scheduler resolve that capability before the model is loaded; it carries no
runtime logic). The driver, hooks, execution plan and runner stay unchanged:
the per-request ragged/chunk-policy opt-in is stamped from the runner.

#### Layout abstraction

The sequence chunking/restoration primitives are model-agnostic and live in
`vllm_omni/diffusion/mixfusion/execution_plan.py` (`PackedExecutionPlan`,
`build_execution_plan`, `gcd_chunk_size`, `plan_chunk_size`,
`split_sequence_chunks`, `reassemble_chunk_rows`, `expand_per_chunk`),
analogous to how `_sp_plan` centralizes sequence-parallel layout. A model
adopts the layout by declaring `_mixfusion_plan`, the MixFusion counterpart
of `_sp_plan`, and does not branch on MixFusion in its forward code:

```python
_mixfusion_plan = {
    "": MixFusionRoot(),
    "image_rope_prepare": MixFusionReplaceOutput(boundary="rope"),
    "transformer_blocks.*.img_mod": MixFusionExpandRows(),
    "norm_out": MixFusionExpandArg(arg_name="conditioning_embedding"),
}
```

- `apply_mixfusion` registers the hooks at the declared module boundaries;
  they stay inert unless the driver arms a `MixFusionStep` on the forward
  context.
- `MixFusionDriver.armed` arms that step for exactly the packed transformer
  call and restores the previous value afterwards (CFG may call the
  transformer twice per step). The root hook then asks the adapter to pack
  the per-request latent list, `MixFusionReplaceOutput` hands `forward` the
  adapter-computed packed value for that boundary (e.g. the packed hidden
  states and RoPE tables), and the expand hooks map per-request conditioning
  rows to per-chunk rows. Without an armed step the model runs its serial
  forward unchanged.
- Attention remains the one layout-aware leaf: the packed layout has
  image-batch = num-chunks versus text-batch = num-requests, so joint
  attention reassembles the per-request logical domain itself instead of
  relying on a shared batch dimension. It reads the plan from the armed
  step instead of a forward argument.
- Paths the packed layout cannot express are refused by the driver's
  admission guards (above); model-specific refusals live in the adapter.
  For Qwen-Image, `can_pack` keeps a batched tensor on the serial forward,
  and `prepare_inputs` raises on `zero_cond_t` (editing). The fused joint
  QK-norm-RoPE table is disabled while a step is armed, because text
  (per-request) and image rows (per-chunk) cannot share one table.

## Model Author Guidelines

When converting a request-level pipeline:

- Reuse the same helpers as `forward()` to avoid behavior drift.
- Copy request-scoped scheduler state into `state.scheduler`.
- Advance `state.step_index` only in `step_scheduler()`.
- Keep model forward work in `denoise_step()`.
- Keep latent mutation in `step_scheduler()`.
- Keep final decoding equivalent to the tail of `forward()`.
- Store masks, condition latents, and other request-local tensors in the
  request state rather than on the shared pipeline.

Before setting `supports_step_execution = True`, validate output parity for the
same seed and sampling parameters, scheduler isolation across concurrent
requests, abort cleanup, progress reporting, and all supported CFG paths.

Before setting `supports_request_batch = True`, validate the batch input
contract, one-output-per-request result shape, request identity and error
mapping, seeded concurrency, LoRA compatibility, and tensor IPC.

## Limitations

- Request batching requires an explicit pipeline capability when
  `max_num_seqs>1`.
- Step execution requires the four-method stateful pipeline contract.
- Both batching paths currently require homogeneous compatibility keys.
- FIFO scheduling can reduce batching opportunities.
- `request_batch_max_wait_ms` trades first-request latency for burst
  coalescing.
- All diffusion cache backends are currently unsupported in step mode.
- KV transfer is supported for newly admitted step requests; some other
  request-mode extras remain unsupported.
- Step continuous batching remains experimental; use `max_num_seqs=1` for the
  conservative step path.
- MixFusion shape relaxation requires the pipeline `supports_mixfusion`
  capability (Qwen-Image only); edit/zero_cond_t requests, SP>1, active
  forward hooks, and `do_true_cfg` fall back to serial ragged execution.

## Related Implementation

- Engine: [`vllm_omni/diffusion/diffusion_engine.py`](gh-file:vllm_omni/diffusion/diffusion_engine.py)
- Model contract: [`vllm_omni/diffusion/models/interface.py`](gh-file:vllm_omni/diffusion/models/interface.py)
- Scheduler interface: [`vllm_omni/diffusion/sched/interface.py`](gh-file:vllm_omni/diffusion/sched/interface.py)
- Request scheduler: [`vllm_omni/diffusion/sched/request_scheduler.py`](gh-file:vllm_omni/diffusion/sched/request_scheduler.py)
- Step scheduler: [`vllm_omni/diffusion/sched/step_scheduler.py`](gh-file:vllm_omni/diffusion/sched/step_scheduler.py)
- Runner: [`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Request batch: [`vllm_omni/diffusion/worker/request_batch.py`](gh-file:vllm_omni/diffusion/worker/request_batch.py)
- Step input batch: [`vllm_omni/diffusion/worker/input_batch.py`](gh-file:vllm_omni/diffusion/worker/input_batch.py)
