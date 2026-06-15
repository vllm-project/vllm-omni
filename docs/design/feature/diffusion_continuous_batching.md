# Continuous Batching for Step-Wise Diffusion

!!! warning "Experimental Feature"
    This feature is experimental. It currently applies to native diffusion
    pipelines running with `step_execution=True`. Heterogeneous dynamic
    batching is implemented for Qwen-Image first.

This document describes the batching extension built on top of
[Diffusion Step Execution](diffusion_step_execution.md). The base
step-execution contract is unchanged: the scheduler admits requests by step,
the runner owns per-request state, and the pipeline still exposes
`denoise_step(input_batch)`, `step_scheduler(...)`, and `post_decode(...)`.

The current implementation supports two batching shapes:

- dense step batching, where requests share a dense tensor layout
- Qwen-Image dynamic heterogeneous batching, where requests may have different
  image token lengths, text lengths, resolutions, and CFG scalar values

## Why It Helps

Step-wise execution breaks a long denoise loop into scheduler-visible units.
That gives the runtime a place to admit other compatible requests between
steps instead of waiting for an entire request to finish.

This matters most in low-MFU or bursty serving scenarios:

- one request's denoise step may not fully saturate the GPU
- several compatible requests can share the same denoise forward pass
- throughput and device utilization can improve without changing
  request-local scheduler state

This is **not** a guaranteed single-request latency win. The main benefit is
higher utilization and better throughput when the workload contains multiple
in-flight compatible requests.

## Current Support

With continuous batching enabled:

- the scheduler may keep multiple compatible requests active at the same time
- the runner packs active request state into one step-local `InputBatch`
- `denoise_step()` runs on that batch
- `step_scheduler()` and `post_decode()` still run per request
- request progress and completion remain independent

For Qwen-Image, the dynamic path additionally supports heterogeneous
co-batching:

- different image resolutions and latent token lengths
- different positive and negative text lengths
- different request-local CFG scalar values, such as `true_cfg_scale`
- different request arrival times, because cached running requests and newly
  admitted requests are rebuilt into the same step view

The Qwen-Image dynamic path is padding-free in attention. The runner builds
varlen `AttentionMetadata` with `padded_tokens=0`, and FlashAttention consumes
flat packed `[total_tokens, heads, head_dim]` Q/K/V tensors.

## Enablement

Use `--step-execution` as the feature gate, then increase `--max-num-seqs`
above `1` if you want batching:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 2 \
  --enforce-eager \
  --cache-backend none
```

`--max-num-seqs 1` keeps the step-wise path without enabling multi-request
batching.

For a reproducible replay flow using the bundled serving benchmark, see the
Qwen-Image replay commands in
[`benchmarks/diffusion/README.md`](gh-file:benchmarks/diffusion/README.md)
and
[`benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md`](gh-file:benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md).

## Scheduler Admission

The scheduler derives batch capacity from `max_num_seqs` through
`max_num_running_reqs`.

Batch admission is gated by
[`SamplingParamsKey`](gh-file:vllm_omni/diffusion/sched/interface.py). The key
is intentionally narrow:

- `do_classifier_free_guidance`
- `lora_int_id`
- `lora_scale`

Spatial shape, temporal shape, text length, and CFG scalar values are not
ordinary scheduler compatibility dimensions. They are request-local state that
the runner interprets when it builds the current execution view.

For Qwen-Image dynamic batching, `do_classifier_free_guidance` is derived from
the actual Qwen CFG execution shape:

- `true_cfg_scale > 1`
- a negative prompt is present

This keeps `true_cfg_scale=1` out of CFG batches even when a negative prompt is
present. Requests with `true_cfg_scale=2` and `true_cfg_scale=7` can share a
batch because they have the same CFG execution shape; their scalar values are
applied per request after positive and negative noise predictions are computed.

Important scheduler details:

- `num_inference_steps` is not part of the key, so requests with different
  total step counts can still share a batch
- active requests do not need the same current denoise progress
- FIFO admission is preserved, so an incompatible request at the head of the
  waiting queue blocks later compatible requests
- LoRA remains a hard key because the worker activates one adapter identity and
  scale for a step batch
- multi-prompt requests do not participate in batching today

Shape removal from the scheduler key does not mean every model can execute
mixed shapes. The current heterogeneous runner path is Qwen-Image-specific.
Models without a dynamic execution view still need dense-compatible tensors or
must run with `max_num_seqs=1`.

## Runner State

The runner keeps persistent per-request execution state in
[`DiffusionRequestState`](gh-file:vllm_omni/diffusion/worker/utils.py). The
scheduler owns a separate lightweight request state for queueing and lifecycle
tracking.

For each step, the runner builds or refreshes an
[`InputBatch`](gh-file:vllm_omni/diffusion/worker/input_batch.py) from active
request states.

The dense view exposes:

- gathered `latents`
- gathered `timesteps`
- padded prompt embeddings and masks
- dense `img_shapes` and text length metadata

The Qwen-Image dynamic view exposes:

- `dynamic_latents`, one request-local latent tensor per request
- `request_spans`, separating request boundaries from segment boundaries
- `txt_seq_lens` and `negative_txt_seq_lens`
- `img_shapes` for request-local RoPE reconstruction
- `true_cfg_scales` and `cfg_normalize_flags`
- `latents=None`, so the dynamic path cannot accidentally use dense latent
  assumptions

`InputBatch.make_batch(..., allow_dynamic=True)` selects the dynamic view when
Qwen-Image can use it and the batch needs request-local layout. Mixed latent
lengths, mixed text lengths, mixed negative text lengths, or mixed CFG scalars
all force the dynamic view. A single request also uses the dynamic view when
enabled, so dynamic-single and dynamic-multi share the same code path.

The current dynamic implementation supports one output row per request.
Dynamic image editing with `image_latents` is rejected for now.

## Attention Metadata

The attention execution contract is
[`AttentionMetadata`](gh-file:vllm_omni/diffusion/attention/backends/abstract.py).
It now carries the varlen fields consumed by attention backends:

- `q_cu_seqlens`
- `kv_cu_seqlens`
- `max_q_len`
- `max_kv_len`
- `q_segments`
- `kv_segments`
- `position`
- `padded_tokens`

[`AttentionSegment`](gh-file:vllm_omni/diffusion/attention/backends/abstract.py)
describes request-local semantic spans, such as `text`, `negative_text`, and
`image`. It is not a cache layout and does not describe LoRA or MoE routing.

For Qwen-Image, the runner builds one metadata entry per attention id:

- `qwen_image.joint.positive`
- `qwen_image.joint.negative`, only when true CFG is active

Each entry contains its own text lengths, image lengths, request order,
segments, and RoPE position plan. The metadata is valid only for the current
step batch composition.

## FlashAttention Varlen

[`FlashAttentionImpl`](gh-file:vllm_omni/diffusion/attention/backends/flash_attn.py)
has a flat varlen branch for packed dynamic Q/K/V tensors:

```text
query/key/value: [total_tokens, heads, head_dim]
metadata: q_cu_seqlens, kv_cu_seqlens, max_q_len, max_kv_len
```

The flat varlen branch rejects attention masks and requires
`padded_tokens=0`. Dense and masked paths remain available for existing dense
execution.

## Qwen-Image Dynamic Path

Qwen-Image consumes the dynamic `InputBatch` inside
[`QwenImagePipeline.denoise_step()`](gh-file:vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py).

The pipeline runs:

1. positive branch prediction using `qwen_image.joint.positive`
2. negative branch prediction using `qwen_image.joint.negative`, if true CFG is
   active
3. per-request CFG combine:

   ```python
   negative + true_cfg_scale * (positive - negative)
   ```

4. optional per-request CFG normalization
5. request-local scheduler step and decode

The transformer dynamic path is implemented in
[`QwenImageTransformer2DModel._forward_dynamic()`](gh-file:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py).
It keeps model-specific work in the model boundary:

- request-local image projection
- request-local text projection
- deterministic RoPE reconstruction from `img_shapes` and text lengths
- per-request timestep/modulation/norm/FFN/output projection
- packed joint attention only where varlen attention is needed
- split of attention output back into text and image streams

This preserves the dense Qwen-Image path for dense tensor inputs.

## Guardrails

The runner fails fast when a dynamic Qwen-Image batch is requested under an
unsupported engine profile:

- not `QwenImagePipeline`
- `step_execution=False`
- `max_num_seqs <= 1`
- `enforce_eager=False`
- diffusion cache backend enabled
- diffusion KV cache dtype enabled
- sequence parallelism enabled
- ring attention enabled
- CFG parallelism enabled
- HSDP enabled
- attention backend is not FlashAttention
- dynamic image editing `image_latents` are present

These are engine/profile constraints, not scheduler batch keys. They are
checked at the runner boundary before the dynamic forward runs.

## Execution Flow

The step-wise batched path is:

1. Run `prepare_encode()` for newly admitted requests.
2. Build or refresh `InputBatch`.
3. Build `dict[attention_id, AttentionMetadata]`.
4. Enter forward context with the metadata map.
5. Run one `denoise_step(input_batch)`.
6. Split dynamic noise predictions by request, or slice dense noise by row.
7. Run per-request `step_scheduler()`.
8. Run `post_decode()` only for completed requests.
9. Remove completed request state from the runner cache.

Dense batches still scatter updated latent tensors back through
[`scatter_latents()`](gh-file:vllm_omni/diffusion/worker/input_batch.py).
Dynamic batches update request-local latents directly through each request's
`step_scheduler()`.

## Validation

The current test coverage includes:

- scheduler key tests for LoRA, CFG shape, resolution removal, and CFG scalar
  removal
- request and segment boundary tests for Qwen-Image metadata
- FlashAttention flat varlen numerical comparison against per-request
  attention
- Qwen-Image transformer dynamic-vs-dense tests
- a deterministic DiT mock-attention test proving dense single, dynamic
  single, and dynamic multi use the same model semantics
- runner stepwise tests for request-local latents, CFG scales, and shape
  metadata
- an online serving e2e test covering:
  - `512x512 + 512x512`
  - `512x512 + 1024x1024`
  - `512x512 + 256x256`
  - staggered arrival
  - different request-local `true_cfg_scale` values
  - dynamic single, dynamic multi, and execute-model reference comparison

## Current Limitations

- Experimental feature; use `max_num_seqs=1` for the older conservative path.
- Heterogeneous dynamic execution is implemented for Qwen-Image first.
- Request-mode diffusion still clamps `max_num_seqs` back to `1`.
- Dynamic batches currently support one output row per request.
- Multi-prompt requests are not batched.
- Qwen-Image edit/image-latent dynamic batching is not supported yet.
- Diffusion cache, diffusion KV cache, CFG parallel, sequence parallel, ring,
  HSDP, and CUDA graph dynamic execution are not supported yet.
- Non-Qwen models need their own runner/model dynamic view before they can
  safely execute mixed shapes.

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
- Attention metadata:
  [`vllm_omni/diffusion/attention/backends/abstract.py`](gh-file:vllm_omni/diffusion/attention/backends/abstract.py)
- FlashAttention backend:
  [`vllm_omni/diffusion/attention/backends/flash_attn.py`](gh-file:vllm_omni/diffusion/attention/backends/flash_attn.py)
- Qwen-Image pipeline:
  [`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`](gh-file:vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py)
- Qwen-Image transformer:
  [`vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py`](gh-file:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py)
- Scheduler tests:
  [`tests/diffusion/test_diffusion_scheduler.py`](gh-file:tests/diffusion/test_diffusion_scheduler.py)
- Dynamic Qwen-Image tests:
  [`tests/diffusion/test_qwen_image_dynamic_step_batching.py`](gh-file:tests/diffusion/test_qwen_image_dynamic_step_batching.py)
- Online serving e2e:
  [`tests/e2e/online_serving/test_qwen_image_dynamic_step_batching.py`](gh-file:tests/e2e/online_serving/test_qwen_image_dynamic_step_batching.py)
