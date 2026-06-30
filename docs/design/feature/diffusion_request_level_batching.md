# Request-Level Batching for Diffusion

This document describes the request-mode batching path for diffusion pipelines.
For end-user enablement and tuning, see
[Request-Level Batching](../../user_guide/diffusion/request_batching.md).

This is separate from
[Continuous Batching for Step-Wise Diffusion](diffusion_continuous_batching.md).
Request-level batching runs one full pipeline `forward()` over a static batch of
compatible requests. Step-wise continuous batching admits work between denoise
steps when `step_execution=True`.

## Why It Helps

The request-level design avoids coupling several logical requests to one request
object. This keeps request identity, abort/error handling, and per-request
metadata unambiguous while still allowing one fused pipeline forward pass for
bursty or concurrent traffic.

## Overview

With request-level batching enabled:

- each `OmniDiffusionRequest` contains one `prompt` and one `request_id`
- the scheduler groups compatible waiting requests into one scheduler wave
- `DiffusionRequestBatch` wraps the scheduled requests for pipeline `forward()`
- batch-capable pipelines return `list[DiffusionOutput]`, one output per
  request
- `BatchRunnerOutput` maps each result back to its original `request_id`

Pipelines opt in with `supports_request_batch = True` and a `forward()` method
that accepts `DiffusionRequestBatch` and returns `list[DiffusionOutput]`.
Pipelines that do not opt in keep the existing per-request execution path.

## Enablement

Request-level batching is the request-mode path, so `step_execution` must remain
disabled. Increase `max_num_seqs` above `1` to let the scheduler keep multiple
compatible requests active:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4
```

For bursty online ingress, `request_batch_max_wait_ms` can add a bounded
admission wait before the first `schedule()` of a scheduler wave:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4 \
  --request-batch-max-wait-ms 20
```

`request_batch_max_wait_ms=0` disables this wait and is the default.

## Request Contract

`OmniDiffusionRequest` represents one logical request. It owns one prompt,
sampling parameters, request id, and request-local metadata. Runtime batches are
formed by the scheduler and represented separately from the request payload.

Runtime batching is represented by:

- [`DiffusionSchedulerOutput`](gh-file:vllm_omni/diffusion/sched/interface.py)
  for scheduled request ids and request payloads
- [`DiffusionRequestBatch`](gh-file:vllm_omni/diffusion/worker/request_batch.py)
  for the pipeline-facing request batch
- [`BatchRunnerOutput`](gh-file:vllm_omni/diffusion/worker/utils.py) for
  per-request results

`DiffusionRequestBatch` intentionally exposes compatibility properties such as
`prompts`, `sampling_params`, `request_id`, and `kv_sender_info` so migrated
pipelines can stay close to upstream code while using a batch-aware contract.

## Scheduler

The scheduler derives its capacity from `max_num_seqs` through
`max_num_running_reqs`. It exposes waiting/running queue counters so the engine
can decide whether admission wait is useful before scheduling a new wave.

Batch compatibility is controlled by
[`SamplingParamsKey`](gh-file:vllm_omni/diffusion/sched/interface.py). The key
contains shape-sensitive and guidance-sensitive fields, including output count
and LoRA identity. Requests with incompatible shapes, CFG settings, output
counts, LoRA adapters, or LoRA scales are kept in separate batches.

Admission is conservative:

- the scheduler only batches compatible requests
- FIFO ordering is preserved
- an incompatible request at the head of the waiting queue blocks later
  compatible requests

## Engine

[`DiffusionEngine`](gh-file:vllm_omni/diffusion/diffusion_engine.py) resolves
request-batch capability during initialization from the configured pipeline
class, including custom pipeline classes.

The capability check uses the pipeline class attribute
`supports_request_batch = True`. Pipelines that set this attribute must implement
a request-batch-compatible `forward()` contract and return one
`DiffusionOutput` per request; the runner validates that return shape at runtime.

When the selected pipeline is batch-capable and `step_execution=False`, request
mode routes scheduler waves through the batch executor path. Otherwise it keeps
the per-request executor path.

The optional admission wait runs only when:

- request batching is supported
- `step_execution=False`
- `request_batch_max_wait_ms > 0`
- no requests are currently running

The wait exits early when the waiting queue reaches capacity, when the queue is
stable for a short window, when the deadline expires, or when the engine stops.

## Executor And Runner

The executor exposes two request-mode entries:

- `execute_request`: one worker call per scheduled request
- `execute_batch`: one worker call for the whole `DiffusionSchedulerOutput`

On the batch path, the worker builds a `DiffusionRequestBatch` and runs the
pipeline once. Request-local setup remains per request:

- KV transfer metadata
- random generator and seed handling
- request output/error/abort mapping

Shared batch setup happens once per batch when possible:

- cache refresh
- LoRA activation for the homogeneous adapter key
- pipeline `forward(req_batch)`

Large tensor IPC still uses the shared-memory packing path. The packer traverses
both normal `RunnerOutput.result` wrappers and nested batch results so batched
outputs do not fall back to pickle IPC for tensor payloads.

## Current Limitations

- Only pipelines that declare the request-batch contract use fused batch
  execution.
- Batches are homogeneous under `SamplingParamsKey`; heterogeneous resolution or
  incompatible guidance settings do not co-batch yet.
- FIFO scheduling can reduce batching opportunities when an incompatible
  request is at the front of the queue.
- `request_batch_max_wait_ms` improves burst coalescing but can add latency to
  the first request in a scheduler wave. Keep it small for latency-sensitive
  serving.
- Step-wise continuous batching is documented separately and only applies when
  `step_execution=True`.

## Related Files

- Request object and request batch:
  [`vllm_omni/diffusion/request.py`](gh-file:vllm_omni/diffusion/request.py)
- Scheduler interface:
  [`vllm_omni/diffusion/sched/interface.py`](gh-file:vllm_omni/diffusion/sched/interface.py)
- Scheduler base:
  [`vllm_omni/diffusion/sched/base_scheduler.py`](gh-file:vllm_omni/diffusion/sched/base_scheduler.py)
- Engine:
  [`vllm_omni/diffusion/diffusion_engine.py`](gh-file:vllm_omni/diffusion/diffusion_engine.py)
- Worker runner:
  [`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Executor interface:
  [`vllm_omni/diffusion/executor/abstract.py`](gh-file:vllm_omni/diffusion/executor/abstract.py)
- Tests:
  [`tests/diffusion/test_diffusion_engine.py`](gh-file:tests/diffusion/test_diffusion_engine.py)
