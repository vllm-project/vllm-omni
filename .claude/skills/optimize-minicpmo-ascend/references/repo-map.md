# vLLM-Omni MiniCPM-o 4.5 Optimization Map

Use this map to locate the current implementation before proposing changes.
Paths may move; use `rg --files` and `rg` to confirm them in the active branch.

## Model and Deployment Baseline

- `recipes/OpenBMB/MiniCPM-o-4_5.md`
  - Known-good serving modes, architecture, performance examples, output
    contracts, and operational notes.
- `vllm_omni/deploy/minicpmo_4_5.yaml`
  - Single-device three-stage layout and NPU platform overrides.
- `vllm_omni/deploy/minicpmo_4_5_batching.yaml`
  - Throughput-oriented split layout; use as a pattern, not as an official
    Ascend competition topology.
- Other `vllm_omni/deploy/minicpmo_4_5*.yaml` files
  - Alternative stage placements and experimental duplex configuration.

Do not assume GPU memory ratios or CUDA graph behavior transfer directly to
the official Ascend environment.

## Three-Stage Pipeline

- `vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py`
  - Pipeline registration and stage wiring.
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py`
  - Thinker/Talker wrapper, multimodal preprocessing, stage behavior, and NPU
    compatibility paths.
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_llm.py`
  - Thinker-side LLM integration.
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py`
  - Talker codec-token generation and request-local state.
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py`
  - Code2Wav stage orchestration and streaming waveform output.
- `vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py`
  - Exact-shape-compatible Code2Wav batching and cache/state handling.
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_token2wav.py`
  - Token-to-wave adapter with device selection and NPU-aware behavior.
- `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`
  - Thinker-to-Talker and Talker-to-Code2Wav data flow, chunk metadata, and
    terminal flush semantics.

Metric ownership is approximate:

| Surface | Primary competition impact |
| --- | --- |
| Thinker preprocessing/prefill/decode | TTFT, text E2E, text throughput, effect |
| Talker generation/scheduling | first audio response, chunk readiness, concurrency |
| Code2Wav/vocoder | audio first packet, chunk latency, continuity, audio E2E |
| Connectors/orchestrator/API | queueing, transfer gaps, global E2E, stability |

## Ascend Platform Layer

- `vllm_omni/platforms/npu/platform.py`
  - Platform registration and capability behavior.
- `vllm_omni/platforms/npu/worker/`
  - NPU model runners, generation runners, workers, and execution flow.
- `vllm_omni/platforms/npu/models/`
  - Model/operator-specific Ascend patches. Reuse established patterns for
    dtype, layout, supported kernels, and compile fallbacks.
- `vllm_omni/platforms/npu/quant/`
  - NPU quantization integration currently present in the repository.
- `vllm_omni/platforms/npu/profiler.py`
  - `torch_npu.profiler` wrapper. Analyze output offline with
    `torch_npu.profiler.profiler.analyse()`.
- `.claude/skills/vllm-omni-npu-upgrade/`
  - Existing workflow for keeping NPU runners aligned with vLLM-Ascend.

When changing shared code, verify GPU behavior is preserved. Prefer an NPU
platform patch when the behavior is Ascend-specific.

## Serving and Benchmark Surfaces

- `examples/online_serving/minicpmo/README.md`
  - Current server and request examples, text/audio separation, Daily-Omni
    notes, streaming, and duplex behavior.
- `examples/online_serving/minicpmo/`
  - Multimodal, streaming, concurrent, and realtime clients.
- `vllm_omni/entrypoints/cli/benchmark/`
  - `vllm bench serve --omni` integration and CLI arguments.
- `vllm_omni/benchmarks/data_modules/daily_omni_dataset.py`
  - Daily-Omni multimodal request construction.
- `vllm_omni/benchmarks/data_modules/daily_omni_eval.py`
  - Daily-Omni effect/accuracy calculation.
- `benchmarks/tts/`
  - TTS serving metrics such as TTFT, E2E, audio TTFP, audio RTF, throughput,
    concurrency sweeps, and streaming underrun. These are useful local proxies
    but are not automatically the official competition benchmark.
- `tests/dfx/perf/scripts/run_benchmark.py`
  - Existing performance sweep patterns and result comparison.

Prefer the official competition benchmark as soon as it is published. Reuse
repository tools for diagnosis and regression coverage, not to redefine the
score.

## Focused Validation

- `tests/model_executor/models/minicpmo_4_5/`
  - Pipeline, Talker, Code2Wav batching, audio processing, and model tests.
- `tests/model_executor/stage_input_processors/test_minicpmo_4_5_*.py`
  - Stage bridge, chunk, and metadata behavior.
- `tests/examples/online_serving/test_minicpmo_streaming.py`
  - Streaming client/output behavior.
- `tests/e2e/online_serving/` MiniCPM-o scenarios
  - Duplex, multi-session, and live serving coverage.
- `tests/platforms/npu/`
  - NPU-specific regression patterns.

Match tests to the risk:

- Config-only tuning: config validation plus Ascend smoke/performance run.
- Scheduling/chunk changes: request-state, boundary, terminal flush, and
  concurrent-request tests.
- Kernel/dtype/layout changes: shape/dtype parity, effect output, and NPU trace.
- Quantization: model load, task accuracy/effect, multimodal/audio quality,
  memory, latency, and throughput.

## Common Bottleneck Questions

1. Is TTFT dominated by API/media preprocessing, Thinker prefill, graph
   compilation, scheduling, or text decode?
2. Does Talker emit the first codec chunk early enough, and does batching delay
   it at higher concurrency?
3. Is Code2Wav compute slower than real time, or are connector/queue gaps the
   real cause of chunk delay?
4. Are chunk shapes fragmenting batches or causing graph recompilation?
5. Are host-device synchronizations, tensor conversions, copies, or temporary
   allocations visible in the NPU trace?
6. Does increased concurrency improve aggregate throughput while making TTFT,
   chunk tails, or stability unacceptable?
7. Does a precision change preserve official effect and multimodal capability?
