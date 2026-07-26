# MiniCPM-o 4.5 Ascend Competition TODO

Status snapshot: 2026-07-25

Competition: [MiniCPM & Ascend Inference Optimization and Application Innovation Challenge](https://ascend.openbmb.cn/competition)

This document tracks the work required to use vLLM-Omni for the competition's
high-performance inference track. New official rules, benchmark scripts, and
starter kits override assumptions recorded here.

## Competition Goal

Treat the work as a gated multi-objective optimization:

1. Pass model effect, correctness, functional, and stability validation.
2. Reduce TTFT, single-chunk latency, and end-to-end latency.
3. Increase throughput and stable concurrent sessions.
4. Improve NPU utilization and memory efficiency.
5. Reproduce the result in the official Ascend environment.

Do not invent metric weights before the organizer publishes the formal scoring
formula.

## Current Pipeline

```text
OpenAI chat request: text / image / audio / video
                         |
                         v
Stage 0: Thinker -> text + hidden states -> final text output
                         |
                         v
                    llm2tts bridge
                         |
                         v
Stage 1: Talker -> request-local autoregressive codec deltas
                         |
                         v
        collect 25 new codec frames + 3 left-context frames
                         |
                         v
Stage 2: Code2Wav -> Flow -> CFM -> HiFT -> 24 kHz audio chunks
```

Relevant implementation:

- Pipeline topology: `vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py`
- Default deployment: `vllm_omni/deploy/minicpmo_4_5.yaml`
- Thinker/Talker wrapper: `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py`
- Talker: `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py`
- Stage bridges: `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`
- Code2Wav: `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py`
- Batched vocoder: `vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py`
- NPU-safe Token2Wav facade: `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_token2wav.py`

## Current Readiness

| Area | Status | Main gap |
| --- | --- | --- |
| Thinker on NPU | Partial | Generic NPU runner plus temporary input-shape workaround; no competition baseline |
| Talker on NPU | Partial | Request-local batching exists; no real NPU concurrency/effect validation |
| Code2Wav on NPU | Blocked | Production path still loads external `stepaudio2.Token2wav` with CUDA assumptions |
| Correctness/effect | Partial | Generic Daily-Omni support exists; no MiniCPM competition gate |
| Performance benchmark | Missing | No competition-specific TTFT/chunk/E2E/concurrency harness |
| NPU observability | Partial | NPU profiler exists; no unified competition report collection |
| Reproduction package | Missing | No official-environment deployment bundle and performance report |

## P0: Competition Blockers

### P0-1 Freeze Official Rules and Environment

- [ ] Download the latest official starter kit and record its URL, date, version, and checksum.
- [ ] Record the official NPU model, card count, topology, CPU/RAM limits, and network policy.
- [ ] Record the official image, driver, firmware, CANN, Python, PyTorch, torch-npu, vLLM, vLLM-Ascend, and vLLM-Omni versions.
- [ ] Record the required MiniCPM-o 4.5 model source, revision, checksum, and model packaging rules.
- [ ] Record the official request schema, datasets, metric definitions, aggregation method, concurrency, timeouts, and effect thresholds.
- [ ] Record package size, execution time, cache, quantization, dependency, and submission limits.
- [ ] Update this document whenever official material changes.

Acceptance:

- [ ] A checked-in environment manifest contains every version and checksum needed for reproduction.
- [ ] No benchmark or score calculation depends on an unpublished assumption.

### P0-2 Make Code2Wav NPU-Native

Current blockers:

- `MiniCPMO45Code2Wav.load_weights()` imports the external
  `stepaudio2.Token2wav` implementation.
- The in-tree `MiniCPMO45Token2wav` facade documents that the external package
  hard-codes `.cuda()`, but the production Code2Wav path does not use the
  in-tree facade.
- `BatchedToken2Wav` expects direct Flow/HiFT modules and cache attributes that
  the current facade does not expose.
- NPU-specific Step-Audio2/HiFT/DiT patches exist, but the MiniCPM production
  path must be wired to the compatible in-tree core.

TODO:

- [ ] Decide the supported NPU backend contract for batched MiniCPM Code2Wav.
- [ ] Replace the CUDA-only external loading path with an in-tree, device-aware backend.
- [ ] Expose or adapt the Flow, HiFT, prompt feature, cache, dtype, and timestep interfaces needed by `BatchedToken2Wav`.
- [ ] Ensure the Step-Audio2 NPU HiFT and DiT attention patches are applied on the actual MiniCPM path.
- [ ] Remove hard-coded CUDA autocast usage from prompt extraction.
- [ ] Add NPU-aware fp32/fp16/bf16 autocast and validate supported operator dtypes.
- [ ] Verify model parameters, buffers, noise tensors, prompt features, and request caches are placed on the intended NPU.
- [ ] Verify temporary reference-audio files and prompt feature caches are released after success, failure, abort, and timeout.
- [ ] Preserve the existing GPU path and output contract.

Acceptance:

- [ ] Text-plus-audio serving starts without importing a CUDA-only runtime.
- [ ] One text request produces valid text and non-empty 24 kHz audio on Ascend.
- [ ] Image, audio, and video input requests produce valid text and audio outputs.
- [ ] Multiple audio chunks are emitted in order and reconstruct one valid waveform.
- [ ] Request-local Flow/HiFT state is isolated under concurrent requests.
- [ ] No NPU memory leak remains after repeated request completion and cancellation.

### P0-3 Establish an Ascend End-to-End Baseline

- [ ] Add an official-environment server startup script.
- [ ] Add deterministic text-only and text-plus-audio smoke requests.
- [ ] Add image, audio, and video input smoke requests.
- [ ] Fix model revision, sampling parameters, seed, output length, and modality settings.
- [ ] Record warmup count and exclude initialization/graph compilation from formal measurements.
- [ ] Save server logs, client results, audio artifacts, error counts, and NPU monitoring output.
- [ ] Run at least one clean restart and reproduce the same functional result.

Acceptance:

- [ ] The full three-stage pipeline runs in the official or closest available Ascend environment.
- [ ] Text-only requests skip Talker and Code2Wav as intended.
- [ ] Text-plus-audio requests exercise all three stages.
- [ ] Baseline commands are runnable without machine-local undocumented state.

### P0-4 Implement the Competition Benchmark Harness

The repository already provides Daily-Omni and generic TTS benchmark
infrastructure, but MiniCPM-o 4.5 is not registered in the universal TTS model
matrix and there is no competition-specific benchmark suite.

- [ ] Implement the official request schema and benchmark invocation.
- [ ] Measure TTFT or first response according to the official definition.
- [ ] Measure first audio response if it is part of the official TTFT definition.
- [ ] Measure single-chunk latency and chunk arrival jitter.
- [ ] Measure E2E latency.
- [ ] Measure request/token/audio throughput according to the official units.
- [ ] Sweep the official concurrent-session configurations.
- [ ] Track p50/p95/p99 or the exact official statistics.
- [ ] Record NPU utilization, peak HBM, host memory, error rate, and timeout rate.
- [ ] Keep profiler runs separate from formal score runs.
- [ ] Save raw JSON results and emit a stable summary table.

Acceptance:

- [ ] One command runs warmup, benchmark, resource collection, and result export.
- [ ] Failed or truncated requests cannot be counted as successful performance samples.
- [ ] Text-only and text-plus-audio results are reported separately.
- [ ] Every reported number is traceable to raw output and an exact command.

### P0-5 Build the Effect and Correctness Gate

- [ ] Integrate the official public effect dataset and evaluation script when released.
- [ ] Add MiniCPM-o 4.5 Daily-Omni coverage as a local proxy until the official suite is available.
- [ ] Fix chat template settings, output modalities, sampling, and answer extraction.
- [ ] Validate text correctness for text, image, audio, and video inputs.
- [ ] Validate speech is non-empty, finite, correctly sampled, and not truncated or duplicated.
- [ ] Add streaming continuity and chunk-order checks.
- [ ] Add concurrent-request state isolation checks.
- [ ] Establish baseline self-run variance before evaluating precision changes.

Acceptance:

- [ ] A failing effect or stability check blocks performance acceptance.
- [ ] The gate produces a machine-readable pass/fail result.
- [ ] The same gate runs before and after every optimization candidate.

## P1: Performance Work After P0 Passes

### P1-1 Remove Talker-to-Code2Wav Host Synchronization

The async bridge currently converts codec tensors to CPU Python scalars before
forming each chunk. This can synchronize NPU execution on every codec delta.

- [ ] Measure the synchronization cost in an NPU trace.
- [ ] Keep codec accumulation in tensor/device-aware storage where possible.
- [ ] Move to host only at the connector boundary if the connector requires it.
- [ ] Avoid rebuilding Python lists and tensors for every generated codec token.
- [ ] Preserve request routing, left context, terminal flush, abort, and epoch semantics.

Acceptance:

- [ ] Single-chunk latency and Talker-to-Code2Wav handoff time improve outside run variance.
- [ ] Chunk contents and final audio remain equivalent to the baseline.

### P1-2 Optimize Code2Wav Dtype and Operators

- [ ] Establish fp32, fp16, and bf16 support per Code2Wav submodule on Ascend.
- [ ] Keep fp32-only S3Tokenizer/HiFT operations in their required precision.
- [ ] Measure Flow encoder, CFM estimator, and HiFT independently.
- [ ] Verify the NPU-safe HiFT downsample and DiT attention paths on real hardware.
- [ ] Remove avoidable layout conversions, copies, allocations, and synchronizations.
- [ ] Cache static timelines, masks, windows, and other shape-stable tensors.

Acceptance:

- [ ] The selected dtype improves performance or memory without failing the effect gate.
- [ ] No CPU fallback dominates the measured chunk path unless explicitly justified.

### P1-3 Improve Stage-2 Batching and Graph Execution

- [ ] Measure exact-shape bucket occupancy at each concurrency.
- [ ] Quantify fragmentation from prompt identity, codec length, cache shape, terminal status, and epoch.
- [ ] Test initial chunk size and steady-state chunk size separately.
- [ ] Tune `codec_chunk_frames` and `codec_left_context_frames` against TTFT, chunk latency, E2E, and quality.
- [ ] Evaluate static-buffer ACL Graph capture for supported exact shapes.
- [ ] Keep an eager fallback for unsupported dynamic shapes.
- [ ] Measure batching gains against queue delay and tail latency.

Acceptance:

- [ ] Throughput improves without unacceptable TTFT/chunk p95 regression.
- [ ] All chunk boundaries, terminal flushes, and request caches remain correct.

### P1-4 Remove the Thinker NPU Batch Workaround

- [ ] Reproduce why the current NPU path squeezes batched inputs.
- [ ] Align Thinker input shapes with the active vLLM-Ascend runner contract.
- [ ] Validate multimodal positions, embeddings, and hidden-state shapes for batch size greater than one.
- [ ] Add NPU-specific batch tests instead of retaining an undocumented shape conversion.

Acceptance:

- [ ] Batched Thinker requests run without the temporary workaround.
- [ ] TTFT, correctness, and multimodal outputs remain stable at concurrency greater than one.

### P1-5 Tune Deployment and Concurrency

- [ ] Test official-card stage placement candidates.
- [ ] Tune per-stage memory utilization and leave measured runtime headroom.
- [ ] Tune `max_num_seqs`, `max_num_batched_tokens`, request admission, and timeouts.
- [ ] Compare colocated and separated Thinker/Talker/Code2Wav layouts if multiple NPUs are allowed.
- [ ] Measure inter-stage transport, queue wait, HCCL cost, and process contention.
- [ ] Run a long-duration stability test at the target concurrent-session count.

Acceptance:

- [ ] The selected layout is Pareto-competitive for latency, throughput, memory, and stability.
- [ ] No stage OOM, starvation, cache leak, or request timeout appears in the stability run.

### P1-6 Add Competition Observability

- [ ] Emit per-stage queue, compute, transfer, and output timing.
- [ ] Emit per-request first-text, first-codec, first-audio, and completion timestamps.
- [ ] Track chunk count, size, cadence, and underrun/jitter indicators.
- [ ] Track actual execution batch size rather than request-scoped placeholders.
- [ ] Collect NPU utilization, HBM, host memory, error count, and restart count.
- [ ] Add a report generator for baseline-versus-candidate comparisons.

Acceptance:

- [ ] A global E2E regression can be attributed to a specific stage or queue.
- [ ] Formal benchmark mode keeps observability overhead low and documented.

## P2: High-Risk Optimizations

Start these only after P0 and the relevant P1 measurement work is complete.

### P2-1 Operator and Kernel Optimization

- [ ] Rank NPU operators by total time and repeated launch count.
- [ ] Optimize only operators proven hot for official shapes.
- [ ] Prefer supported CANN/Ascend kernels and platform-local patches.
- [ ] Add shape, dtype, numerical, effect, and performance validation for every custom kernel.

### P2-2 Quantization and Precision Reduction

- [ ] Confirm official rules and packaging requirements for quantized weights or runtime quantization.
- [ ] Establish unquantized effect and performance baselines first.
- [ ] Evaluate Thinker, Talker, and Code2Wav quantization independently.
- [ ] Measure accuracy, multimodal behavior, speech quality, latency, throughput, and memory.
- [ ] Reject any candidate that fails the official effect threshold or has inconclusive quality.

### P2-3 Algorithmic Changes

- [ ] Confirm whether altered sampling, speculative methods, approximation, or caching is permitted.
- [ ] Keep benchmark-specific behavior out of model logic.
- [ ] Document the mechanism and validate against hidden-distribution risk.

## Submission Deliverables

- [ ] Reproducible source code and configuration.
- [ ] Environment manifest with versions and checksums.
- [ ] Server startup, warmup, correctness, benchmark, profiling, and monitoring scripts.
- [ ] Raw benchmark results and summarized performance report.
- [ ] Effect/correctness report and reviewable output artifacts.
- [ ] Deployment and reproduction instructions for the official Ascend environment.
- [ ] Package-size and runtime-limit checks.
- [ ] Clean-environment reproduction log.
- [ ] Known limitations, tradeoffs, and rollback instructions.

## Experiment Record Template

Use one record per candidate:

| Field | Value |
| --- | --- |
| Candidate ID | |
| Git SHA / diff | |
| Official rules/toolkit version | |
| Environment and NPU topology | |
| Model revision and dtype | |
| Server command | |
| Benchmark command | |
| Workload and concurrency | |
| Hypothesis | |
| Baseline metrics | |
| Candidate metrics | |
| Effect/correctness result | |
| NPU utilization / peak HBM | |
| Stability result | |
| Raw artifact paths | |
| Decision: keep/reject/investigate | |

## Recommended Execution Order

1. P0-1 official rules and environment.
2. P0-2 NPU-native Code2Wav.
3. P0-3 end-to-end Ascend baseline.
4. P0-4 benchmark harness and P0-5 effect gate.
5. P1-1 host synchronization and P1-2 Code2Wav dtype/operator work.
6. P1-3 batching/graph, P1-4 Thinker batch support, and P1-5 topology tuning.
7. P1-6 observability and repeated baseline validation.
8. P2 work only when profiling and competition rules justify it.
