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
Qwen-Image, HunyuanImage3, and Helios. Step execution alone does not imply
continuous-batching support: Qwen-Image accepts batched step states.
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

## Related Implementation

- Engine: [`vllm_omni/diffusion/diffusion_engine.py`](gh-file:vllm_omni/diffusion/diffusion_engine.py)
- Model contract: [`vllm_omni/diffusion/models/interface.py`](gh-file:vllm_omni/diffusion/models/interface.py)
- Scheduler interface: [`vllm_omni/diffusion/sched/interface.py`](gh-file:vllm_omni/diffusion/sched/interface.py)
- Request scheduler: [`vllm_omni/diffusion/sched/request_scheduler.py`](gh-file:vllm_omni/diffusion/sched/request_scheduler.py)
- Step scheduler: [`vllm_omni/diffusion/sched/step_scheduler.py`](gh-file:vllm_omni/diffusion/sched/step_scheduler.py)
- Runner: [`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Request batch: [`vllm_omni/diffusion/worker/request_batch.py`](gh-file:vllm_omni/diffusion/worker/request_batch.py)
- Step input batch: [`vllm_omni/diffusion/worker/input_batch.py`](gh-file:vllm_omni/diffusion/worker/input_batch.py)
