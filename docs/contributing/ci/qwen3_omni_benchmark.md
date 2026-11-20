# Qwen3-Omni dataset accuracy and performance

Qwen3-Omni uses the shared `tests/dfx/perf/scripts/run_benchmark.py` runner
with `tests/dfx/perf/tests/test_qwen3_omni_datasets.json`. Explicit accuracy
cases reuse their inference requests for performance and scoring. Accuracy
is independent of concurrency. Every non-duplex request generates text and audio.

The two combined CUDA jobs have 120-minute timeouts. Their labels include both
`Accuracy Test` and `Perf Test`, so `tools/nightly/run_nightly_jobs.sh` selects
them with either the `acc` or `perf` filter and schedules them before performance
report generation. The `function` filter excludes these benchmark jobs.

## Scenario matrix

| Platform | Deployment | Dataset | Concurrency | Measured requests | Accuracy |
| --- | --- | --- | --- | --- | --- |
| CUDA | default | English Seed-TTS | 64 | 1088 | yes |
| CUDA | async-prefix | Daily-Omni | 8 | 1197 | yes |
| CUDA | async-prefix | English Seed-TTS | 1 / 16 / 32 / 64 | 32 / 256 / 512 / 1088 | at 64 |
| NPU | default | English Seed-TTS | 32 | 1088 | yes |
| NPU | async-prefix | English Seed-TTS | 1 / 16 / 32 | 32 / 256 / 1088 | at 32 |

CUDA has six runs; NPU has four. Seed-TTS accuracy cases use all 1088
utterances. Performance-only tiers use fixed prefixes of the pinned manifest:
32 utterances at c1, 256 at c16 and 512 at CUDA c32. Every tier requires
complete request/audio/metric coverage for its configured sample count.
Daily-Omni uses all 1197 questions, including both 30-second and 60-second videos. Sources are
pinned by revision and manifest SHA256 in `tests/dfx/perf/qwen3_omni_workload.py`.
Shuffling and oversampling are disabled. Offline mirrors can be selected with
`QWEN3_OMNI_DAILY_ROOT` and `QWEN3_OMNI_SEED_ROOT`; their original manifests
must match the checksums.

Both deployments use MRv2. Default disables async chunk and all stage prefix
caches. Async-prefix enables async chunk and Thinker prefix caching only.
CUDA uses two inference devices: Thinker TP1 on device 0, Talker and Code2Wav
share device 1. CUDA Seed-TTS evaluators reuse device 1 after measurement.
NPU inherits the production topology and memory limits: Thinker TP2 on devices
0 and 1, Talker and Code2Wav share device 2, and device 3 runs the evaluators.
These IDs identify software Devices. Each A3 physical NPU contains two Devices
(Dies); selecting one ID does not select both Dies or combine their memory.
Production deploy YAMLs are not changed; JSON applies temporary test overrides.
NPU scenarios retain production PIECEWISE graph mode, per-stage token budgets,
tensor parallelism, placement and memory limits, while selecting the requested
async-chunk and prefix-cache policies.
Benchmark servers allow 1200 seconds per stage and 1800 seconds for overall
initialization, covering cold compilation before requests are measured.
Daily-Omni and Seed-TTS use separate server groups. A failed media engine therefore
does not get reused for Seed's concurrency sweep; Seed tiers share one lifetime.

The full-duplex JSON configurations use the same shared runner. CUDA covers
Omni-DuplexEval `PR_correction` (120 selected IDs, c8) and OmniInteract
`1q1a_math` (60 sessions, c8). The NPU DuplexEval candidate covers the same
120 IDs at c8. CUDA uses device 2 for the judge; NPU uses device 3. Candidate NPU profiles
require qualification before replacing scheduled A3 jobs. A2/H200 supporting
results identify their actual hardware and do not establish A3/H100 results.

## Completeness, accuracy and performance

Every dataset run requires its exact submitted/completed/audio count, no
request failures or timeouts, and complete positive finite performance samples.
Accuracy cases also require complete scorer coverage and finite quality scores.
Invalid Daily answers count as wrong answers; they do not become missing
requests. Shared AMD and legacy accuracy helpers retain their behavior.

Quality thresholds are Daily accuracy >= 0.69; Seed-TTS WER <= 0.05 and
SIM >= 0.75. UTMOS scoring is disabled on both platforms.
WER/SIM scoring follows the request measurement window. Performance-only
Seed-TTS cases explicitly disable these evaluators. The two existing
verbatim-reading cases belong to the online function expansion suite.

Each run performs `max(2, concurrency)` warmup requests outside measurement.
`--omni-isolate-prefix-cache` gives measured requests a fresh shared cache salt;
readiness and warmups use separate salts. Reuse inside a measured group remains
possible. Multimodal processor caches retain existing behavior. The benchmark
measures a warmed service. Use the recorded case order when comparing revisions.

TTFT, TTFP, TPOT, E2E, audio duration and request throughput are retained.
Daily performance references use TTFT/RTF/TTFP; Seed-TTS references use RTF/TTFP.
Throughput is an aggregate metric, not a per-request percentile.
`tests/dfx/perf/baselines/qwen3_omni.json` starts in `collect` mode. It enforces
metric validity immediately and defers numerical performance acceptance until
repeated target-hardware calibration. References must identify the hardware,
case, dataset revision, request count, reference commit, at least two repeated
measurements, metric values and relative tolerances. Set
`QWEN3_OMNI_BASELINE_MODE=enforce` or use `--require-baseline` after calibration.

The shared runner persists one `result_test_*.json` for each dataset case,
including quality, performance, software/data revisions, qualification scope
and validation errors, before asserting success. HTML/Excel readers consume
these ordinary performance artifacts. Performance subsets are marked
`qualification_scope: performance-subset`; accuracy cases are `full-dataset`.
Performance references must match the exact request count; subset and
full-dataset measurements are separate workloads. Unmeasurable TPOT is null.

Evaluated DuplexEval cases also publish `result_acc_*.json` under
`tests/e2e/accuracy/omni_duplex_eval/results` (or `ACC_BENCH_RESULT_DIR`). This
artifact copies the existing judge scores and coverage from the performance
result; inference and judging run once. The CUDA DuplexEval job uploads both
result directories. Publication does not require `duplex_eval_accuracy_config`;
record-only cases retain an empty accuracy baseline and their existing score
checks. The performance JSON, metric definitions and numerical scores are
unchanged.

## Small smoke runs before full experiments

Copy selected JSON entries into a separate smoke configuration. Set a small
`num_prompts`, reduce `num_warmups`, and set
`qwen3_omni_workload.smoke` to true. Accuracy-enabled smoke runs still require
all selected requests to have valid audio and complete finite scores, but do
not apply full-dataset quality or calibrated performance floors. Their results
are marked `qualification_scope: smoke` and cannot qualify a full dataset.
Daily smoke can use explicit `smoke_indices`, for example `[0, 647]`, to
exercise both clip durations. Other runs use the source manifest order.

Accuracy qualification rejects truncated request counts. Performance subsets
are regular measured cases, with their own sample counts and performance
references. A successful smoke must precede each deployment's scheduled
experiment. Do not continue after startup,
audio, scoring or result-publication errors without resolving the failure.

## Running

With matching vLLM/Omni dependencies, model, media and evaluators cached:

```bash
BENCHMARK_DIR=tests/dfx/perf/results \
SEED_TTS_EVAL_DEVICE=cuda:1 \
pytest -sv tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_datasets.json \
  -m 'full_model and cuda and cards_2' --run-level full_model
```

For the NPU candidate, use `SEED_TTS_EVAL_DEVICE=npu:3`,
`SEED_TTS_SIM_DEVICE=npu:3`, and `-m 'full_model and npu and cards_4'`.
Run inside the host's GPU/NPU scheduler. Full-duplex uses
`test_qwen3_omni_duplex_omni_duplex_eval.json` with `cuda and cards_3` or
`npu and cards_4`. CPU contracts run without loading model dependencies:

```bash
pytest --noconftest -o addopts='' tests/benchmarks/test_qwen3_dataset_contract.py
```
