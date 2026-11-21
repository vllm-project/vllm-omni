# Qwen3-Omni dataset accuracy and performance

Qwen3-Omni uses the shared `tests/dfx/perf/scripts/run_benchmark.py` runner
with `tests/dfx/perf/tests/test_qwen3_omni_non_full_duplex.json`. Explicit accuracy
cases reuse their inference requests for performance and scoring. Accuracy
is independent of concurrency. Daily-Omni takes video and audio as input and
requests a text answer. Seed-TTS requests both text and audio output.

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
pinned by revision and manifest SHA256 in each scenario's `dataset_source`.
Shuffling and oversampling are disabled. Offline mirrors can be selected with
`QWEN3_OMNI_DAILY_ROOT` and `QWEN3_OMNI_SEED_ROOT`; their original manifests
must match the checksums.

The shared DFX parameter fixture resolves `dataset_source` and scopes
`benchmark_env`. Daily-Omni and Seed-TTS materialize their source prefixes in
their existing dataset modules, before loading can skip invalid rows. Every
case enters the ordinary `run_benchmark.py` sweep and shared result writer.
`result_validation` declares accuracy, coverage requirements and quality
thresholds; `result_metadata` declares deployment labels and device counts.
The model, backend, input packing and Seed-TTS system prompt are explicit JSON
parameters, so another model can reuse the same path with its own settings.

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
`1q1a_math` (60 sessions, c8). A3 nightly runs DuplexEval on the same 120 IDs
at c8, with the judge on device 3. CUDA uses device 2 for the judge.
The four non-duplex NPU cases still await A3 qualification. A2/H200 supporting
results identify their actual hardware and do not establish A3/H100 results.

## Completeness, accuracy and performance

Every dataset run requires the configured number of submitted and completed requests,
zero request failures or timeouts, and complete positive finite performance samples.
Seed-TTS also requires audio output for every request.
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

Daily-Omni records TTFT, TPOT, ITL, E2E latency and request throughput.
The text-only case retains the H100 mean TTFT reference of 3647.318585 ms
from the two earlier runs with text and audio output. Seed-TTS also records audio
duration and uses audio RTF and TTFP for performance regression checks.
Throughput is an aggregate metric, not a per-request percentile.
`result_validation.metrics` checks that each metric has a valid value and
the expected number of samples. The H100 references in
`benchmark_params[].baseline` use the arithmetic mean for each scenario in
[build 20463](https://buildkite.com/vllm/vllm-omni/builds/20463), running
[commit `9d86d17f12`](https://github.com/vllm-project/vllm-omni/commit/9d86d17f1223f77c713b04dd398c24257b6c53e5).
Default Seed-TTS uses three runs, and each async-prefix scenario uses two.
Both CUDA nightly jobs pass `--assert-baseline` and fail if a calibrated
H100 metric exceeds its reference by more than 10%. B200 has no calibrated
reference and continues to collect measurements.

The H100 full-duplex reference is the arithmetic mean of each day's
response-level `mean_duplex_busy_rtf` on November 17, 18 and 20, 2026, after
the [Talker CUDA graph fix](https://github.com/vllm-project/vllm-omni/pull/10499).
The three runs give 0.1262 for OmniInteract and 0.1749 for DuplexEval.
See [build 20013](https://buildkite.com/vllm/vllm-omni/builds/20013),
[build 20145](https://buildkite.com/vllm/vllm-omni/builds/20145) and
[build 20463](https://buildkite.com/vllm/vllm-omni/builds/20463).

For A3 DuplexEval, the two completed runs in [NPU build 12466](https://buildkite.com/vllm/vllm-omni-npu-ci/builds/12466)
measured response-level mean busy RTFs of 0.716089 and 0.733625 at merge commit `541da19a71`.
Both evaluated all 120 `PR_correction` samples at concurrency 8. Their mean, 0.724857,
is the `A3` reference. This case maps the runtime label `910` reported in both
calibration files to `A3` for baseline selection and result publication.

The judge scored 89/120 (74.17%) on each run, with zero skipped or failed
evaluations. The accuracy artifacts report aggregate scores for this split.
The three H100 runs above each scored 90/120. For both H100 and A3, nightly
requires `pr.mean_all_success >= 0.70` (at least 84/120). This allows five
fewer successes on A3 and six on H100 relative to the calibration runs.
The accuracy artifact records the selected floor in `baseline.min_pr_mean_all_success`.
The numerical performance check uses mean busy RTF. The other measurements below are recorded for comparison.

| A3 DuplexEval metric | [First run](https://buildkite.com/vllm/vllm-omni-npu-ci/builds/12466#01a1fa05-f45f-45e7-8c50-511a9e51735d) | [Second run](https://buildkite.com/vllm/vllm-omni-npu-ci/builds/12466#01a1fb8d-e346-4894-a80d-66c93b09168d) | Relative change |
| --- | ---: | ---: | ---: |
| Request throughput (requests/s) | 0.170779 | 0.170158 | −0.36% |
| Mean client RTF | 0.777752 | 0.794242 | +2.12% |
| Mean audio RTF | 0.771014 | 0.788487 | +2.27% |
| Mean stall ratio | 0.053411 | 0.053518 | +0.20% |
| Mean client TTFT (ms) | 529.631 | 576.200 | +8.79% |
| Mean client TTFP (ms) | 812.461 | 884.290 | +8.84% |
| Mean response TTFT (ms) | 400.083 | 449.960 | +12.47% |
| Mean response TTFP (ms) | 661.450 | 738.439 | +11.64% |
| P95 response TTFT (ms) | 520.642 | 688.189 | +32.18% |
| P95 response TTFP (ms) | 954.779 | 1427.107 | +49.47% |
| P99 response TTFT (ms) | 751.679 | 1636.654 | +117.73% |
| P99 response TTFP (ms) | 1072.009 | 2058.277 | +92.00% |

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
result; inference and judging run once. The CUDA and A3 DuplexEval jobs upload both
result directories. Publication does not require `duplex_eval_accuracy_config`;
record-only cases retain an empty accuracy baseline and their existing score
checks. The performance JSON, metric definitions and numerical scores are
unchanged.

## Small smoke runs before full experiments

Copy selected JSON entries into a separate smoke configuration, omitting the
performance `baseline` field. Set a small
`num_prompts`, reduce `num_warmups`, and set
`result_validation.smoke` to true. Accuracy-enabled smoke runs require complete finite scores for all selected
requests. Seed-TTS smoke also requires valid audio for each request. Full-dataset
quality and calibrated performance floors apply to qualification runs. Their results
are marked `qualification_scope: smoke` and cannot qualify a full dataset.
Daily smoke can use explicit `dataset_source.sample_indices`, for example `[0, 647]`, to
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
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_non_full_duplex.json \
  -m 'full_model and cuda and cards_2' --run-level full_model
```

For the NPU candidate, use `SEED_TTS_EVAL_DEVICE=npu:3`,
`SEED_TTS_SIM_DEVICE=npu:3`, and `-m 'full_model and npu and cards_4'`.
Run inside the host's GPU/NPU scheduler. Full-duplex uses
`test_qwen3_omni_duplex_omni_duplex_eval.json` with `cuda and cards_3` or
`npu and cards_4`. CPU contracts run without loading model weights:

```bash
pytest --noconftest -o addopts='' tests/benchmarks/test_qwen3_dataset_contract.py
```
