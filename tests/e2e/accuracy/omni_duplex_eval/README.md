# Omni-DuplexEval CI Guard

The nightly runs **66 samples at concurrency 8**, generates responses once,
then evaluates those same responses. Performance timing ends before judging.
The DFX runner writes both performance and accuracy artifacts.

## Configuration and execution

| File | Responsibility |
| --- | --- |
| `tests/dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json` | Model, judge, benchmark parameters and hardware marks |
| `omni_duplex_eval_ci_config.json` | Expected dataset, protocol pin, sample IDs and quality thresholds |
| `validation.py` | Coverage, protocol, score-file and RTD/PR quality checks |
| `test_omni_duplex_eval_ci.py` | CPU regressions for the accuracy contract |
| `tests/dfx/conftest.py` | Common result metadata and artifact publication |

Run from the repository root:

```bash
export BENCHMARK_DIR=tests/dfx/perf/results
export ACC_BENCH_RESULT_DIR=tests/e2e/accuracy/omni_duplex_eval/results
pytest -s -v tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json \
  -m "full_model and cuda and (H100 or B200) and cards_2" \
  --run-level full_model
```

For A3, use `-m "full_model and npu and A3 and cards_2"`.
The existing CUDA and NPU nightly jobs use this same JSON configuration.
Setting `duplex_eval_accuracy_config` opts a DFX case into the accuracy
artifact and CLI-exit guard. Existing evaluated cases such as Qwen3-Omni's
`pr_correction` keep their current result and exit-code behavior.

The model uses the first allocated device and the judge uses the second.
DFX maps the judge's `device_index` through the platform's visible-device
environment. The judge is `Qwen/Qwen2.5-VL-7B-Instruct`, with a 65,536-token
context, video input at 2 FPS, and four evaluation workers. Its allowed
media directory matches the configured response root.

Input replay uses the real-time clock and 1 FPS. Warmups and request-based
readiness probes are disabled; the server fixture waits for startup.
MiniCPM uses its bundled `assets/HT_ref_audio.wav`. The benchmark parameters
and accuracy reference must select the same 66 IDs: 30 RTD and 36 PR samples.

## Results

Performance and accuracy reports use their respective result directories.
`BENCHMARK_DIR` selects the performance directory; `ACC_BENCH_RESULT_DIR`
selects the accuracy directory, following the existing MiniCPM accuracy tests.
The defaults are:

```text
tests/dfx/perf/results/
├── result_<test>_<hardware>_<dataset>_...json
└── omni_duplex_eval_minicpmo_4_5/run-*/
    └── ... detailed audio, transcript, event and judge artifacts

tests/e2e/accuracy/omni_duplex_eval/results/
└── result_acc_<model>_omni_duplex_eval_<date>_<time>.json
```

Accuracy filenames follow the shared Daily-Omni / Seed-TTS convention, for
example `result_acc_MiniCPM-o-4_5_omni_duplex_eval_20261020_064320.json`.
Hardware and concurrency are recorded in the JSON. `model_id` and `date`
retain the benchmark metadata, `dataset` identifies the evaluation suite, and
`dataset_path` records its Hub repository or local data path.
`name` records the optional benchmark configuration label (`null` when unnamed),
and `num_prompts` records the requested sample count.

The accuracy artifact uses this envelope:

```json
{
  "test_name": "test_minicpmo_4_5_omni_duplex_eval",
  "name": null,
  "model_id": "openbmb/MiniCPM-o-4_5",
  "dataset": "omni_duplex_eval",
  "dataset_path": "Hothan/Omni-DuplexEval",
  "date": "20261020-065052",
  "Hardware": "H100",
  "baseline": {
    "min_rtd_mean_content_score": 1.8,
    "min_rtd_mean_temporal_score": 1.4,
    "min_pr_mean_all_success": 0.28
  },
  "branch": "main",
  "num_prompts": 66,
  "max_concurrency": 8,
  "benchmark_exit_code": 0,
  "generation": {
    "completed": 66,
    "failed": 0,
    "published": 66,
    "published_with_unfinished_responses": 0,
    "artifact_errors": []
  },
  "accuracy": {
    "status": "completed",
    "protocol_pin": "ca3c122b4d4bf67afd6b18ea5e724b4561bdde48",
    "samples": 66
  }
}
```

The `generation` object preserves publication coverage and artifact errors.
The full `accuracy` object also contains RTD/PR scores, per-task results,
coverage counts, errors and the score directory. `status` describes judge
completion; the subsequent assertions enforce the recorded quality thresholds.
Both result artifacts are saved before the assertions, preserving partial
results when judging fails or a score falls below its threshold. Buildkite
uploads JSON files from both directories and detailed artifacts even on test failure.
The CLI must exit successfully; the existing NPU SIGABRT-on-shutdown exception
(`-6`) remains accepted when all artifact and score checks pass.

## Accuracy metrics and fixed thresholds

RTD means **Real-Time Description**: content and temporal scores each use
a 0–3 scale. PR means **Proactive Reminder**: `mean_all_success` is the
fraction of samples for which every required event succeeded, on a 0–1 scale.

| Metric | Minimum passing score | H800/C8 reference, 2026-10-20 |
| --- | ---: | ---: |
| RTD mean content score | 1.8 | 2.0167 |
| RTD mean temporal score | 1.4 | 1.6667 |
| PR mean all-success rate | 0.28 | 11/36 = 0.3056 |

These fixed thresholds use the H800/C8 run at commit `2308562c` as the
reference, with the same judge, protocol and 66 IDs. The RTD thresholds allow
approximately 10.7% and 16% below that run's content and temporal scores.
For the 36 PR samples, a threshold of 0.28 requires at least 11 successes:
10/36 is approximately 0.2778, so one fewer success than this reference fails.
Later runs compare against these checked-in
values; changing the thresholds requires an explicit configuration update.
The CUDA and NPU jobs share these thresholds. H100/B200 and NPU validation,
repeated-run stability checks, and hardware-specific performance baseline
calibration remain pending.

Keep per-task results when interpreting the aggregate. The reference's PR
correction score was 11/13; event reminder and post-event reminder each scored
zero. Passing the aggregate establishes the stated aggregate threshold. Separate
measured task baselines can track improvements in those two tasks.

The guard checks complete generation, publication and evaluation; exact
score-file count and split/ID membership; matching protocol pins in the
configuration, runtime, summary and each score file; all nine task groups;
and finite, in-range RTD/PR scores meeting the configured thresholds. The accuracy
suite keeps its existing scoring eligibility: silent output and responses
still active at session close contribute their collected content to the judge.
The batch must include at least one session with a measured client audio
packet (`client_ttfp_ms.count > 0` and a finite mean). This catches a complete
audio outage even when text answers pass the judge, while individual silent
sessions remain eligible. The saved H800 reference has audio in all 66 sessions.
An audio-coverage ratio floor can be calibrated from repeated runs.
Artifacts retain unfinished-response counts and missing latency measurements,
so performance comparisons can account for response coverage.
Changes to the judge, protocol or selected samples call for fresh references.

## Timing interpretation

The job uses the shared benchmark's server and client latency metrics.
See [benchmark timing definitions](../../../../docs/cli/bench/serve.md)
for their measurement origins.

Server TTFT and Audio TTFP record the first text and audio output independently.
When both first appear in one model output, they share an observation timestamp.
This occurred in all 66 sessions of the H800 reference, giving equal means of
277.978 ms. These describe the response output boundary; Thinker token timing
is available separately in the per-response `stage0_tokens` metrics.

Session E2EL and request throughput include real-time input replay.
Audio RTF retains the benchmark's documented generation-window-to-output-
duration definition. Compare like-for-like workloads, hardware and clock origins.

## CPU regression tests

```bash
pytest -q tests/e2e/accuracy/omni_duplex_eval/test_omni_duplex_eval_ci.py \
  tests/dfx/perf/tests/test_runner_metadata.py \
  tests/benchmarks/test_duplex_session_metrics.py
```

These use synthetic score files and stubbed subprocesses to exercise the real
summarizer, validation, DFX CLI assembly and result writer. GPU execution uses
the DFX command above.
