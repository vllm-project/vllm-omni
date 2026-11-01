# Async Omni Output Materialization

## Table of Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Architecture](#architecture)
4. [Enabling the Feature](#enabling-the-feature)
5. [Compatibility and Fallbacks](#compatibility-and-fallbacks)
6. [Related Files](#related-files)

## Overview

Async Omni output materialization moves CPU-side construction of
`OmniModelRunnerOutput` out of the autoregressive (AR) decode critical path.
It complements [Async Chunk](async_chunk.md): async chunk pipelines partial
outputs between stages, while async output materialization prevents the
construction of those partial outputs from blocking the next decode step.

An AR stage must return sampled token IDs quickly because the scheduler needs
them for the next decode step. However, an Omni stage can also produce hidden
states, multimodal tensors, streaming inter-stage payloads, and connector
metadata. Building these payloads can require device-to-host (D2H) copies,
tensor slicing, flattening, and Python object construction.

Without this feature, all of that work runs inline in `sample_tokens()`:

```text
sample tokens
  -> update decode state
  -> copy hidden and multimodal outputs to CPU
  -> build per-request payloads
  -> build the streaming wire payload
  -> collect connector signals
  -> return sampled tokens
  -> launch the next decode step
```

With async output materialization, only work required by the next decode step
stays on the critical path:

```text
sample tokens
  -> update decode state
  -> snapshot output state and start asynchronous D2H copies
  -> register sampled tokens
  -> launch the next decode step

background output path
  -> wait for the payload snapshot
  -> build per-request payloads
  -> build the streaming wire payload
  -> collect connector signals
  -> construct OmniModelRunnerOutput
```

This allows payload construction for step `N` to overlap with GPU decode work
for step `N + 1`. It does not change model computation or generated output.

!!! note "Platform and validation scope"
    The performance and correctness validation in this document covers CUDA
    and ROCm. This is a validation scope, not a runtime platform guard: XPU
    inherits `GPUARModelRunner`, and MUSA selects `GPUARWorker`, so those
    backends may enter the asynchronous path when its other guards pass. XPU
    and MUSA have not been validated for this optimization. Ascend NPU uses a
    separate `NPUARModelRunner` path that still constructs
    `OmniModelRunnerOutput` and drains connector output synchronously before
    wrapping the result as an asynchronous runner output.

<p align="center">
  <img
    alt="Decode step gap before and after async output materialization"
    src="../../../source/architecture/qwen3-omni-async-output-step-gap.svg"
    width="100%"
  >
</p>

<p align="center">
  <em>
    Figure 1: Moving payload construction off the decode path reduces the
    observed gap between consecutive Talker steps from about 2.8 ms to 41 µs.
    Adapted from the
    <a href="https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization">
      Qwen3-Omni optimization blog
    </a>.
  </em>
</p>

For Qwen3-Omni, the feature is enabled for both AR stages:

- **Thinker**: Defers hidden-state and multimodal payload materialization.
- **Talker**: Runs its decode-state postprocess eagerly, then defers codec
  payload materialization. Because Code2Wav consumes codec codes rather than
  Talker hidden states, the Talker also skips unnecessary hidden-state D2H.
- **Code2Wav**: Does not use this path because it is a generation stage rather
  than an AR stage.

Qwen3-TTS uses the same mechanism in its AR Talker stage. The Talker updates
the decode state it needs for the next step before returning, then constructs
the codec payload for Code2Wav in the background.

## Performance

In the controlled Qwen3-Omni optimization sweep, enabling async output
materialization on top of CUDA Graph and async chunk produced the following
results at concurrency 64:

| Configuration | Request throughput | Mean audio TTFP | Mean audio RTF |
| --- | ---: | ---: | ---: |
| CUDA Graph + async chunk | 9.3 req/s | 655 ms | 0.63 |
| + async output materialization | 11.3 req/s | 631 ms | 0.47 |
| Change | **+22%** | **-4%** | **-25%** |

The sweep used `Qwen/Qwen3-Omni-30B-A3B-Instruct`, 640 Seed-TTS English
prompts, and one replica each for Thinker, Talker, and Code2Wav. Async output
materialization recovers throughput by removing CPU payload construction from
the decode critical path without regressing time to first audio packet (TTFP).
See the
[Qwen3-Omni optimization blog](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization)
for the complete benchmark methodology and results.

The implementation was also validated against the synchronous safe-copy path.
The text and audio output hashes matched, confirming that overlap does not
change the generated result. See
[PR #4476](https://github.com/vllm-project/vllm-omni/pull/4476) for the
correctness and profiling details.

## Architecture

### Output Lifecycle

`OmniAsyncGPUModelRunnerOutput` extends vLLM's
`AsyncGPUModelRunnerOutput`. It preserves the existing asynchronous
sampled-token feedback while adding a background builder for the complete Omni
output.

The output lifecycle is:

1. `GPUARModelRunner.sample_tokens()` samples tokens and performs the
   bookkeeping required by the next decode step.
2. The runner snapshots step-local metadata such as request IDs, token spans,
   scheduler output, and query start locations.
3. CUDA payload tensors are cloned before reusable model or CUDA Graph buffers
   can be overwritten.
4. The cloned payload is copied to pinned CPU memory on a dedicated CUDA
   stream. A CUDA event records when the copy is ready.
5. `OmniAsyncGPUModelRunnerOutput` submits to a runner-owned single-worker executor
   that waits for the payload event and builds `OmniModelRunnerOutput`.
6. The AR runner immediately registers the sampled-token CPU copy with the
   input batch and returns, allowing async scheduling to advance.
7. When the engine calls `get_output()`, it waits for the builder future,
   propagates any builder exception, and finalizes sampled tokens and logprobs
   through the upstream output implementation.

```text
GPU / decode path                         Background output path

forward + sample
      |
      +-- clone output tensors
      +-- enqueue D2H copy --------------> wait for D2H event
      +-- register sampled tokens               |
      +-- return async output                    +-- slice per request
      |                                         +-- build multimodal payloads
next decode step                                +-- partition streaming payloads
                                                +-- build the wire payload
                                                +-- drain connector signals
                                                +-- build OmniModelRunnerOutput
```

The executor reuses its thread across decode steps and processes builders in
submission order. Builder failures propagate through each output future without
stopping subsequent tasks. Runner shutdown drains pending builders before
releasing connector state, CUDA graphs, or tensor buffers. The existing engine
output queue bounds in-flight work; this change does not introduce a new
admission path.

Unlike the previous daemon thread, `ThreadPoolExecutor` workers are joined at
Python interpreter exit. Embedders must call `GPUARModelRunner.shutdown()`
while the runner and device resources are still valid; interpreter exit is not
a substitute for draining output work. The stage engine calls shutdown from
its `finally` block, and upstream worker shutdown delegates to the runner.
If an output builder or device event never completes, graceful shutdown can
still block; a supervisor must forcibly terminate an unresponsive process.
`shutdown(wait=False)` would not remove the interpreter's join requirement.

### Safe State Snapshots

The next scheduler step can mutate runner state while the background builder is
still active. Most step-local builder inputs are therefore detached snapshots.
The snapshotted state includes:

- scheduler token counts and speculative-token metadata
- request IDs and request-to-batch-index mappings
- sampled token IDs, logprobs, and prompt logprobs
- query start locations and scheduled-token spans
- hidden states and multimodal output tensors
- KV-connector and encoder-cache outputs captured for the step

CUDA tensor cloning is required because CUDA Graph and model output buffers can
be reused by the next forward pass. The dedicated copy stream and pinned host
buffers make the D2H transfer asynchronous; the snapshot retains its cloned
CUDA sources until the transfer event completes.

Small lists of multiple compatible CUDA tensors are packed for one snapshot
copy. A singleton list or tuple uses the ordinary tensor clone/copy path,
preserving the container without an unnecessary pack and unpack.

MiniCPM-o 4.5's Talker opts out of hidden-state payloads: Code2Wav consumes
codec codes and metadata only. Its hidden states remain available for logits
and sampling but are not cloned, copied to the host, or transported in the
output payload. The Thinker retains its hidden-state handoff for TTS
conditioning.

### Live Runner State and Ownership

The complete builder input is not a detached snapshot. Connector output state
remains runner-owned: `get_omni_connector_output()` drains live, per-cycle
connector signals after the streaming payload is built. Correctness therefore
depends on that builder being the only consumer that drains connector signals
for its output cycle. Receive-side connector state written by the background
receiver is coordinated by the connector mixin's `_lock`.
`OmniAsyncGPUModelRunnerOutput.get_output()` waits for the builder future and is the
completion and exception boundary before the materialized output is consumed.

The shared output builder also contains a full-payload accumulation branch
that reads `self.requests`. That branch is not reachable when async Omni output
materialization is enabled: this feature requires `async_chunk`, while
`should_accumulate_full_payload_output()` returns `False` whenever
`async_chunk` is enabled. The feature path partitions each step's payload into
streaming inter-stage and client handoffs instead.

### Output Builder

The background builder performs the work that previously ran inline:

- resolves which requests need downstream payloads
- converts hidden and multimodal tensors into per-request CPU payloads
- applies model-specific payload processing
- partitions per-step payloads into inter-stage and client streams
- creates the tensor-only `multimodal_outputs` wire payload
- drains live connector readiness signals after building the streaming payload
- constructs the final `OmniModelRunnerOutput`

Input-side connector operations remain synchronous with model execution. In
particular, receiving inputs and flushing previously completed connector
outputs are not deferred because they affect scheduler-visible request state.

### Qwen3-Omni Stage Behavior

| Stage | Async output behavior | Reason |
| --- | --- | --- |
| Thinker | Snapshots hidden states and multimodal outputs; builds the downstream payload in the background | The Talker needs the payload, but the next Thinker decode step only needs sampled-token feedback |
| Talker | Runs lightweight postprocess eagerly; snapshots codec outputs; omits hidden states from the downstream payload | `hidden_states.last` is needed by the next Talker step, while Code2Wav only needs codec codes |
| Code2Wav | Uses the normal generation-stage output path | Code2Wav is not executed by `GPUARModelRunner` |

The Qwen3-TTS Talker follows the same pattern as the Qwen3-Omni Talker:
postprocess runs eagerly because the next decode step needs its updated state,
while codec payload construction runs asynchronously. Its Code2Wav stage also
uses the normal generation-stage path.

## Enabling the Feature

There is no separate `async_omni_output` command-line or YAML option. The
feature is selected automatically for model stages that opt in when the runtime
conditions are safe. On the validated CUDA and ROCm configurations, using the
bundled deployment profile is enough to turn it on for the supported
Qwen3-Omni and Qwen3-TTS pipelines. The selection logic has no CUDA/ROCm-only
guard; see [Compatibility and Fallbacks](#compatibility-and-fallbacks) for the
other platform paths. The same profile does not enable background Omni
materialization on Ascend NPU.

### Example 1: Qwen3-Omni

Start the full Qwen3-Omni pipeline:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

The model registry automatically loads
`vllm_omni/deploy/qwen3_omni_moe.yaml`. Its relevant settings are:

```yaml
async_chunk: true

stages:
  - stage_id: 0  # Thinker
    enable_prefix_caching: false
    # async_scheduling defaults to true for AR stages

  - stage_id: 1  # Talker
    enable_prefix_caching: false
    # async_scheduling defaults to true for AR stages

  - stage_id: 2  # Code2Wav
    enable_prefix_caching: false
    async_scheduling: false
```

With this profile, async output materialization activates automatically for:

- **Stage 0, Thinker**: Defers hidden-state and multimodal payload
  construction.
- **Stage 1, Talker**: Defers codec payload construction after eagerly
  updating its decode state.

Stage 2 does not use this feature because Code2Wav is not an AR stage.

### Example 2: Qwen3-TTS

Start Qwen3-TTS with any checkpoint that uses the Qwen3-TTS Talker pipeline.
For example:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni \
  --port 8091
```

The model registry automatically loads `vllm_omni/deploy/qwen3_tts.yaml`. Its
relevant settings are:

```yaml
async_chunk: true

stages:
  - stage_id: 0  # Talker
    async_scheduling: true
    enable_prefix_caching: false

  - stage_id: 1  # Code2Wav
    enable_prefix_caching: false
```

Async output materialization activates only for Stage 0, the AR Talker. The
Talker keeps the state required by its next decode step on the GPU, skips an
unnecessary hidden-state D2H copy, and builds the codec payload for Code2Wav in
the background. Stage 1 uses the generation-stage output path.

The same behavior applies to the Base and VoiceDesign Qwen3-TTS checkpoints
because they use the same Talker implementation.

!!! warning
    Do not pass `--no-async-chunk` when you want this optimization. Disabling
    async chunk causes the runner to fall back to synchronous output
    construction. Enabling async chunk alone cannot activate the feature for a
    model that has not opted into async Omni output.

!!! note "Prefix cache compatibility"
    Prefix cache and async Omni output can run together. `save_outputs` returns
    a `step_id`; the output builder later calls `materialize(sid, req_ids)` with
    the save-time request list, which may be a step late relative to the live
    `input_batch`. `_should_use_async_omni_output()` does not disable itself
    when the cache is present. See
    [Automatic Prefix Caching](prefix_caching.md#implementation).

## Compatibility and Fallbacks

`GPUARModelRunner` uses the async output path only when all of the following
runtime conditions hold:

| Requirement | Reason |
| --- | --- |
| AR async scheduling is enabled | The optimization relies on the scheduler advancing while the prior output is materialized |
| `async_chunk` is enabled | The feature targets incremental downstream Omni payloads |
| The model stage opts in with `use_async_omni_output` | Models must declare that their output lifecycle is safe to defer |
| Speculative decoding is disabled | Speculative output state is not included in this deferred path |
| Routed-expert output is disabled | Routed-expert extraction currently requires the synchronous path |
| Postprocess is absent or explicitly runs eagerly | State needed by the next decode step must be updated before the runner returns. Eager postprocess sees the scheduled slice (`None, None` combined tensors), not the prefix-cache-merged full prompt; it must only consume the tail |

These checks are evaluated per stage on runners based on `GPUARModelRunner`.
An unsupported combination does not prevent serving; it only falls back to
synchronous output construction for that stage. CUDA and ROCm are the validated
platforms. XPU inherits `GPUARModelRunner`, and MUSA selects `GPUARWorker`, so
they may select the async path when these checks pass, but this optimization is
not yet validated on either backend.

Ascend NPU does not use this selection path. `NPUARModelRunner.sample_tokens()`
fully constructs `OmniModelRunnerOutput`, calls
`get_omni_connector_output()`, and only then creates the asynchronous wrapper,
so Omni payload materialization remains synchronous.

!!! note
    `use_async_omni_output`,
    `eager_omni_postprocess_before_async_output`, and
    `omni_pooler_payload_include_hidden` are model implementation contracts,
    not user-facing configuration fields.

## Related Files

- `vllm_omni/worker/gpu_ar_model_runner.py`: Async output object, safe tensor
  snapshots, compatibility guards, and deferred Omni output builder.
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`: Separate Ascend NPU
  runner, where Omni output materialization remains synchronous.
- `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`: Thinker and
  Talker opt-in behavior.
- `vllm_omni/deploy/qwen3_omni_moe.yaml`: Default Qwen3-Omni deployment
  settings.
- `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py`: Qwen3-TTS
  Talker opt-in and eager postprocess behavior.
- `vllm_omni/deploy/qwen3_tts.yaml`: Default Qwen3-TTS deployment settings.
- `tests/worker/test_gpu_ar_model_runner.py`: Snapshot, guard, connector
  ordering, and background error-propagation tests.
- [Async Chunk](async_chunk.md): Inter-stage chunking and scheduling design.
- [Automatic Prefix Caching](prefix_caching.md#implementation): Prefix-cache
  `step_id` / `materialize` contract used by the deferred output builder.
- [Qwen3-Omni optimization blog](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization):
  Optimization context and controlled performance results.
- [PR #4476](https://github.com/vllm-project/vllm-omni/pull/4476): Feature
  implementation, validation, and review history.

## Idle generation steps

Async-chunk generation stages can poll for downstream input many times between
model forwards. After the first idle step has cleared the persistent batch,
subsequent zero-token steps with no request or encoder-cache updates skip the
input-preparation event and batch-state refresh. The first idle step, request
transitions, block zeroing or copying, encoder work, speculative decoding, and KV/EC
transfer retain the normal path. This shortcut is restricted to DP=1 and PP=1.
Connector receive/flush, model lifecycle hooks, and idle prefetch still execute;
the optimization does not skip polling or change the engine's idle sleep.
This input-update shortcut applies to `GPUGenerationModelRunner`; the separate
Ascend `NPUGenerationModelRunner` retains its existing input-update path.

MRv2 AR output also honors the model's `omni_pooler_payload_include_hidden`
contract, in conjunction with the sampler's `include_hidden_states` flag.
Opting out omits hidden-state output copies, while logits and sampling continue
to receive the original hidden states. Models without the flag retain the
existing hidden-state payload behavior.

The MRv2 generation runner has its own control-only fast path. Both its
control-only and cached-request zero-token paths deliver pending request
prewarms after finished-request cleanup and run optional idle prefetch only on
consecutive idle steps. Dummy/profile runs do not enqueue or execute prewarms,
but a dummy run with scheduled tokens clears the idle flag so the next real
idle step cannot prefetch while that work may still be in flight.
Warmup uses the forward thread, stream and inference mode; failures log once
per hook at warning level (subsequent failures at debug) and leave the inline
reference preparation path available. Connector and auxiliary-output work
still runs on every idle step.

## NPU generation output copies

`NPUGenerationModelRunner` has a separate path for non-autoregressive outputs
such as Code2Wav waveforms. With asynchronous scheduling and pinned host memory,
it submits D2H copies on the producer NPU stream and returns an
`AsyncModelRunnerOutput`. `get_output()` waits for a completion event before
exposing CPU tensors to the consumer. The copies precede the next forward on
that stream, including graph replay that overwrites static output buffers.
Device sources remain referenced until completion, CPU metadata is snapshotted,
and request mappings are copied before returning.

Synchronous scheduling waits once for the batch of copies. Full-payload
accumulation and execution-time measurement also retain synchronous readiness;
without pinned memory, copies remain blocking. These guards prevent accumulation
or serialization from observing unfinished CPU tensors. This changes where the
host waits, not the amount of NPU model computation or the D2H dependency on it.
A shorter `sample_tokens()` interval alone is not an end-to-end speedup.
