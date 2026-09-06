# Qwen3-Omni and Qwen3-TTS Model Runner V2 Architecture

## Status

This document describes the implementation in this change. It covers the
Qwen3-Omni Thinker, Talker, and Code2Wav stages and the Qwen3-TTS Talker and
Code2Wav stages.

The v1 runner remains available while the v2 path is validated. This document
does not set a removal date for v1 and does not describe unimplemented
abstractions.

## Goals

- Run autoregressive and generation stages on vLLM Model Runner V2.
- Keep model-specific preprocessing and state in `OmniModelState`.
- Transfer stage payloads without rebuilding request state in the orchestrator.
- Preserve v1 sampling, conditioning, streaming, terminal, and abort semantics.
- Batch Talker MTP and Code2Wav work where request shapes permit it.
- Use CUDA graphs only when the model declares a replay-safe output contract.

## Non-goals

- Replacing connector backends.
- Changing model weights or sampling defaults to manufacture throughput gains.
- Claiming tensor-parallel or pipeline-parallel support beyond paths covered by
  distributed tests.
- Removing the v1 runner in this change.

## Architecture

```text
API / Orchestrator
        |
        | request admission and final output only
        v
Omni scheduler
        |
        | SchedulerOutput + request/span lifecycle
        v
OmniARModelRunner / OmniGenerationModelRunner
        |
        +--> OmniModelState
        |      request-resident intermediate state
        |      model-specific preprocess/postprocess
        |      static graph inputs and Talker MTP batches
        |
        +--> OmniRunnerDataPlane
               request snapshots and ordered output worker
               connector payload build and enqueue
               readiness, terminal, abort, and cleanup
                       |
                       v
               next-stage scheduler inbox
```

### Runner boundary

`OmniGPUModelRunner` extends vLLM's v2 `GPUModelRunner` with Omni lifecycle
hooks. `OmniARModelRunner` owns sampling stages. `OmniGenerationModelRunner`
owns non-sampling generation stages such as Code2Wav.

The runners do not reconstruct the legacy v1 input batch. They delegate
request-specific state to `OmniModelState` and stage transport to
`OmniRunnerDataPlane`.

### Model-state boundary

`OmniModelState` extends vLLM's `DefaultModelState`. It owns:

- `OmniIntermediateBuffer` request entries;
- prompt-embed and runtime-additional-information materialization;
- Qwen3-Omni prefill/decode coordinates;
- model preprocess and postprocess hooks;
- static input buffers used by CUDA graph replay;
- Talker MTP batch preparation and result placement.

Model-specific hooks are exposed through `OmniModelStatePlugin`. A model
without a hook uses the default no-op behavior.

### Native data plane

`OmniRunnerDataPlane` owns the transport-side request snapshot. The snapshot
contains only the fields required by stage payload builders, including prompt
and output token history when sampling semantics require it.

Model output follows this path:

1. The runner reserves each request before handing the batch to the ordered
   output worker.
2. The worker updates cumulative request state and builds connector entries.
3. All entries from one model step are passed to `send_chunks()`, allowing a
   model batch hook to perform one batched D2H/materialization operation.
4. The connector enqueue completes before the lifecycle reservation is
   released.
5. A pending terminal is emitted only after all reserved outputs are committed.

The request-state lock and connector send lock use one order:

```text
native_output_lock -> native_send_lock
```

The output worker builds the entry and claims connector send order while it
holds `native_output_lock`. It then releases the request-state lock during the
enqueue, so `request_terminal()` can record a pending terminal without blocking
the scheduler. An abort may update request state during that enqueue, but its
terminal cannot acquire `native_send_lock` until the accepted data enqueue has
completed. Lifecycle reservations are released only after a successful send.

Native sends use `propagate_errors=True`. Both batch and scalar payload
builders must propagate construction failures. Legacy callers retain the
best-effort default.

### Readiness

The connector receive worker publishes lightweight readiness to a scheduler
inbox. Tensor payload ownership remains in the runner/data plane. The scheduler
consumes readiness at the next `schedule()` call and restores
`WAITING_FOR_CHUNK` requests whose payload is materialized.

The output-carried readiness path remains a compatibility fallback where a
direct scheduler sink is unavailable. The two paths must produce the same
request IDs and terminal semantics.

## Model-specific behavior

### Qwen3-Omni

The pipeline consists of:

1. Thinker: multimodal text generation and captured auxiliary hidden layers.
2. Talker: text-conditioned codec generation plus residual codebooks from MTP.
3. Code2Wav: streaming codec-to-waveform decoding.

Thinker-to-Talker metadata includes authoritative prefill/decode coordinates.
Talker conditioning must use these coordinates rather than infer state from
span length. Codec chunks carry absolute ranges so repeated or overlapping
connector updates can be handled idempotently.

Talker MTP is prepared across runnable requests by `OmniModelState`. Static
buffers and graph buckets are used only for graph-safe shapes. Code2Wav batches
ready chunks with bounded padding; graph replay slices padded output back to
each request's authoritative length.

### Qwen3-TTS

The pipeline consists of:

1. Talker: text/reference conditioning and codec generation.
2. Code2Wav: streaming waveform generation.

Reference-audio artifacts are built in the batched prefill hook and are visible
to scalar preprocessing in the same scheduler iteration. The worker-local LRU
stores device tensors so a cache hit does not add D2H/H2D copies.

Only `input_values` crosses the reference-audio preprocessing boundary.
Effective codec length is derived from the CPU waveform length using the
tokenizer's downsampling contract. This is valid only for the unpadded
per-request waveform supplied by the serving path; callers providing padded
batches must also preserve authoritative lengths.

Reference codec context is published at its first decode use. Prompt tokens are
packed and transferred once for the batch. These operations change
materialization and scheduling, not model sampling parameters.

## CUDA graph output contract

Tuple-returning models may use FULL CUDA graph replay only when they declare
`supports_mrv2_full_graph_aux_outputs`.

For a declared model:

- the primary output must be a tensor;
- auxiliary output must be a non-empty tensor-only pytree;
- every auxiliary leaf must have the same leading batch dimension;
- capture records the pytree schema;
- replay must return both the primary tensor and flattened auxiliary leaves;
- replay reconstructs the original pytree before downstream processing.

A tuple-returning model without this contract is restricted to PIECEWISE
capture. The current vLLM graph manager exposes private candidate containers
for this filtering. If those containers are missing or have an unknown shape,
startup fails closed. Continuing with FULL capture could silently discard
Thinker auxiliary layers and is not a performance-only degradation.

## Correctness invariants

The following invariants apply to Qwen3-Omni and Qwen3-TTS:

1. Data accepted before terminal is enqueued before terminal.
2. No data chunk is emitted after terminal or abort cleanup.
3. Payload construction failure does not release a lifecycle reservation.
4. Terminal is emitted once, including duplicate terminal requests.
5. Duplicate or overlapping codec ranges do not duplicate model input.
6. Talker prefill/decode coordinates are monotonic and match scheduler truth.
7. Batched and scalar payload builders are semantically equivalent.
8. CUDA graph padding is removed before request-visible output.
9. Fixed-seed v1/v2 comparisons preserve token/frame and stop behavior within
   the model's numerical determinism boundary.
10. Abort and output completion use the same lock order.

## Validation

The implementation is covered by focused tests in:

- `tests/worker_v2/` for model state, runner execution, graph contracts, and
  native data-plane lifecycle;
- `tests/worker/test_omni_connector_mixin.py` for batch/scalar connector
  behavior and error propagation;
- `tests/core/sched/` for readiness, restore, terminal, and scheduling policy;
- Qwen3-Omni and Qwen3-TTS model tests for conditioning, codec framing, graph
  padding, and reference-audio artifacts;
- online/offline E2E tests for streaming and serving behavior.

Performance acceptance requires an interleaved MRv1/MRv2 run on the same
hardware, model, deploy shape, prompt set, seed policy, warmup, and output
length distribution. Reports must include:

- request throughput;
- generated audio seconds per wall-clock second;
- TTFA ramp and steady-state distributions;
- output-length p50/p95/max;
- peak GPU memory;
- raw benchmark artifacts.

Audio quality acceptance compares the same prompts and references with UTMOS,
SQUIM, WER where applicable, and speaker similarity. Throughput results are not
correctness evidence and quality results are not performance evidence.

## Known limitations

- Model Runner V2 rejects pipeline parallelism and prefill context parallelism
  at runner startup. Use the v1 runner for those deployment shapes.
- Direct readiness delivery requires a scheduler sink in the local engine
  process. Distributed layouts without that binding use the output-carried
  fallback and require separate TP/PP validation.
- Graph-manager filtering depends on private vLLM containers and intentionally
  fails startup when their contract drifts.
- Artifact-cache capacity consumes GPU memory and must be included in peak
  memory measurements.
- The v1 and v2 runners coexist until full CI, distributed, performance, and
  quality evidence is attached to the change.
