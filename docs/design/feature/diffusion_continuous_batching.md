# Continuous Batching for Step-Wise Diffusion

!!! warning "Experimental Feature"
    This feature is experimental. It currently applies only to native
    diffusion pipelines running with `step_execution=True`.

This document describes the batching extension built on top of
[Diffusion Step Execution](diffusion_step_execution.md). The base
step-execution contract is unchanged. The batching work is mainly in the
scheduler and runner layers.

## Why It Helps

Step-wise execution breaks a long denoise loop into scheduler-visible units.
That gives the runtime a place to admit other compatible requests between
steps instead of waiting for an entire request to finish.

This matters most in low-MFU or bursty serving scenarios:

- one request's denoise step may not fully saturate the GPU
- several compatible requests can share the same denoise forward pass
- throughput and device utilization can improve without changing request-local
  scheduler state

This is **not** a guaranteed single-request latency win. The main benefit is
usually higher utilization and better throughput when the workload contains
multiple in-flight compatible requests.

## Overview

With continuous batching enabled:

- the scheduler may keep multiple compatible requests active at the same time
- the runner packs request-local step state into one `InputBatch`
- `denoise_step()` runs on that batch
- `step_scheduler()` and `post_decode()` still run per request

The current implementation is conservative:

- only compatible requests are batched together
- request-mode diffusion still runs with `max_num_seqs=1`
- per-request progress and completion remain independent

## Enablement

Use `--step-execution` as the feature gate, then increase `--max-num-seqs`
above `1` if you want batching:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

`--max-num-seqs 1` keeps the step-wise path without enabling batching.

For a reproducible replay flow using the bundled serving benchmark, see the
Qwen-Image replay commands in
[`benchmarks/diffusion/README.md`](gh-file:benchmarks/diffusion/README.md)
and
[`benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md`](gh-file:benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md).

## Scheduler

The scheduler derives its batch capacity from `max_num_seqs` through
`max_num_running_reqs`.

Batch admission is gated by
[`SamplingParamsKey`](gh-file:vllm_omni/diffusion/sched/interface.py),
which is built from shape-sensitive and CFG-sensitive sampling fields. This is
the core correctness rule for batching: requests are only co-batched when they
share the same denoise tensor contract.

There are two important details:

- `num_inference_steps` is not part of the key, so requests with different
  total step counts can still share a batch
- requests also do not need to be at the same current denoise progress; active
  requests can continue batching even when their current step indices diverge
- admission is still FIFO, so an incompatible request at the head of the
  waiting queue blocks later compatible requests

Today that compatibility rule is still shape-sensitive. `height`, `width`,
`num_frames`, and CFG-related fields remain part of the key, so different
resolutions or incompatible guidance settings do **not** co-batch yet.

The current batching unit is one `OmniDiffusionRequest`. Requests with
multiple prompts do not participate in batching today.

## Runner

The runner keeps persistent per-request execution state in
[`DiffusionRequestState`](gh-file:vllm_omni/diffusion/worker/utils.py),
while the scheduler owns a separate lightweight request state for queueing and
lifecycle tracking.

For each step, the runner builds an
[`InputBatch`](gh-file:vllm_omni/diffusion/worker/input_batch.py) from the
active request states:

- prompt embeddings and masks are normalized and padded
- dynamic tensors such as `latents` and `timesteps` are gathered each step
- buffers are reused when batch composition stays the same

The step-wise batched path is:

1. Run `prepare_encode()` for newly admitted requests.
2. Build or refresh `InputBatch`.
3. Run one batched `denoise_step(input_batch)`.
4. Slice the batched `noise_pred` back per request.
5. Run per-request `step_scheduler()`.
6. Run `post_decode()` only for requests that finished denoising.
7. Scatter updated latents back into persistent request state with
   [`scatter_latents()`](gh-file:vllm_omni/diffusion/worker/input_batch.py).

This keeps the shared work limited to the denoise forward pass while preserving
request-local scheduler state and outputs.

## Engine

[`DiffusionEngine`](gh-file:vllm_omni/diffusion/diffusion_engine.py) provides
the background loop and async add-request path needed for multiple requests to
accumulate in the scheduler.

This is supporting infrastructure, not the main design point. The batching
behavior is defined by scheduler-side compatibility gating and runner-side
batch packing.

## Current Limitations

- Experimental feature; use `max_num_seqs=1` for the older conservative path.
- Only native pipelines that already support `step_execution=True`.
- Request-mode diffusion still clamps `max_num_seqs` back to `1`.
- Only homogeneous batches keyed by `SamplingParamsKey` are supported.
- Multi-prompt requests are not batched.
- `cache_backend`, KV transfer, and other request-mode extras are not wired
  into the batched step-wise path yet.
- Future work can relax the current same-shape restriction with richer
  heterogeneous batching policies such as bucketing or padded execution for
  different resolutions.

## Related Files

- Scheduler base:
  [`vllm_omni/diffusion/sched/base_scheduler.py`](gh-file:vllm_omni/diffusion/sched/base_scheduler.py)
- Scheduler interface:
  [`vllm_omni/diffusion/sched/interface.py`](gh-file:vllm_omni/diffusion/sched/interface.py)
- Step scheduler:
  [`vllm_omni/diffusion/sched/step_scheduler.py`](gh-file:vllm_omni/diffusion/sched/step_scheduler.py)
- Runner:
  [`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Input batch:
  [`vllm_omni/diffusion/worker/input_batch.py`](gh-file:vllm_omni/diffusion/worker/input_batch.py)
- Tests:
  [`tests/diffusion/test_diffusion_scheduler.py`](gh-file:tests/diffusion/test_diffusion_scheduler.py)
