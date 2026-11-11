# Qwen3-Omni non-duplex nightly

The CUDA and Ascend nightly suites use Daily-Omni and English Seed-TTS to
measure accuracy and serving performance. A concurrency-64 run supplies both
sets of metrics from the same inference requests. ASR, speaker similarity,
UTMOS and answer scoring run after the benchmark measurement window.

## Workloads

All requests generate text and audio. The default deployment explicitly
disables async chunk and prefix caching. The async-prefix deployment enables
async chunk and Thinker prefix caching; audio-stage prefix caching stays off.
Both inherit the production runner and hardware placement.

| Configuration | CUDA concurrency / requests | Ascend A3 concurrency / requests |
| --- | --- | --- |
| default | 64 / full dataset | 64 / full dataset |
| async-prefix | 64 / full dataset | 64 / full dataset |
| async-prefix, performance only | 1 / 32, 16 / 256, 32 / 512 | 1 / 32, 4 / 64, 8 / 128 |

Each row applies to both datasets. There are ten benchmark runs per hardware
profile, four of which also evaluate accuracy. The CUDA jobs use two H100 or
two B200 devices. The A3 job reserves four devices: Thinker uses devices 0–1,
audio stages use device 2, and the accuracy evaluators use device 3.

Daily-Omni uses all 1197 questions at revision
`bf5a6ee4c829510b7a14869e3104efeafc330b07`. English Seed-TTS uses all 1088
utterances at revision `8f5e1aa2a35d42f42e940074c1983358b9491f89`.
The source manifests are SHA256-verified, with shuffling and oversampling
disabled. Smaller runs use a manifest truncated to the first N entries before
the dataset loader runs, so missing early entries cannot be replaced by later
ones. Changing these sources requires recalibrating the performance references.

The benchmark resolves these exact snapshots from the Hugging Face cache.
Offline agents must stage them before running. Alternatively, set
`QWEN3_OMNI_DAILY_ROOT` to a mirror containing the original `qa.json` and either
`Videos/` or `Videos.tar`; set `QWEN3_OMNI_SEED_ROOT` to a mirror containing
`en/meta.lst` and the corresponding audio files. Reformatting either manifest
changes its checksum. The Daily video archive is large; staging and extraction
are outside the performance measurement window.

## Validation and metrics

Each run must submit and complete exactly its expected number of requests and
return decodable audio with positive duration, TTFP and RTF for every request.
The full runs also require all expected questions or utterances to participate
in scoring. High mean scores cannot conceal failed HTTP requests, missing
audio, missing source rows or partial evaluator coverage.

The existing quality thresholds remain: Daily accuracy at least 0.69, Seed-TTS
WER at most 0.05 and SIM at least 0.75. CUDA also requires UTMOS at least 3.10.
A3 does not run UTMOS because its offline runner lacks the evaluator assets.
Two small verbatim-reading tests remain separate CUDA functional regressions.

Daily-Omni still requests only an A/B/C/D answer. Its accuracy denominator and
answer extraction remain unchanged; incomplete execution is a separate failure.
Its short audio workload should not be compared directly to long Seed-TTS
utterances. When there are no measurable text intervals, TPOT is recorded as
null rather than zero.

Each run warms up with `max(2, concurrency)` requests, outside the measurement
window. `--omni-isolate-prefix-cache` isolates its measured KV prefixes from
earlier runs and warmups. Natural reuse within one measured run remains enabled
for async-prefix. Multimodal processor caches retain their existing behavior;
this suite measures a warmed service, not cold startup. Keep the case order
stable when comparing revisions.

## Establishing performance references

`tests/e2e/accuracy/qwen3_omni/nightly_baselines.json` starts in **collect** mode
with no numeric references. Request completeness, metric validity and accuracy
are enforced immediately. Numeric performance regression enforcement requires
calibration on each target hardware profile; local H200 measurements do not
substitute for H100, B200 or A3 references.

Repeat the same workload on a fixed reference commit. Retain the hardware and
software versions, data revision, deployment and individual result artifacts.
Determine each reference value and tolerance from the repeated measurements.
Reference entries are indexed by hardware and case ID, for example
`H100` → `async-prefix-daily-omni-c16`, with this structure:

```text
dataset_revision: the pinned dataset revision
num_prompts: 256
reference_commit: the measured commit
repetitions: number of repeated measurements (at least two)
metrics:
  mean_ttft_ms: {value: measured reference, relative_tolerance: measured tolerance}
  mean_audio_rtf: {value: measured reference, relative_tolerance: measured tolerance}
  mean_audio_ttfp_ms: {value: measured reference, relative_tolerance: measured tolerance}
```

Daily uses all three metrics; Seed uses the two audio metrics. For each metric,
the allowed maximum is `value * (1 + relative_tolerance)`. Other metrics are
retained for diagnosis. References do not update automatically.

After qualification, set the manifest mode to `enforce`. The environment
variable `QWEN3_OMNI_BASELINE_MODE=enforce` allows testing enforcement before
changing the default. Missing hardware/case references, mismatched datasets or
request counts, invalid metrics, and regressions fail the test.

The unified `result_acc_*.json` artifacts include `nightly_suite`, configuration,
hardware, concurrency, source revision, quality and performance metrics, and
validation errors. The repository's HTML and Excel performance readers accept
these combined artifacts. There is no second inference pass or duplicated
performance result. External dashboard consumers may need to adopt the same
suite discriminator. The smaller performance-only runs use `result_test_*.json`
and set `accuracy_evaluated` to false. AMD and weekly configuration files remain
unchanged.

## Running and qualification

In a complete vLLM-Omni uv environment with the model, dataset media and scoring
models cached, the CUDA CI command is:

```bash
uv run --no-sync pytest -sv \
  tests/e2e/accuracy/qwen3_omni/test_qwen3_omni_nightly.py \
  -m 'full_model and cuda and H100 and B200 and cards_2' --run-level full_model
```

On this shared development host, wrap model tests in `gpu run` and pin the
scorers as in CI:

```bash
gpu run --gpus 2 --nonblock --timeout 6h --note 'Qwen3 nightly qualification' -- \
  env SEED_TTS_EVAL_DEVICE=cuda:1 SEED_TTS_UTMOS_DEVICE=cuda:1 \
  uv run --no-sync pytest -sv \
  tests/e2e/accuracy/qwen3_omni/test_qwen3_omni_nightly.py \
  -m 'full_model and cuda and cards_2' --run-level full_model
```

For A3, set `SEED_TTS_EVAL_DEVICE=npu:3` and select
`-m 'full_model and npu and A3 and cards_4'`. Use `-k default` or
`-k async-prefix` to match the two CI jobs. These commands do not run against
AMD. The CPU contract tests require pytest, PyYAML and the report dependencies:

```bash
uv run --no-sync pytest --noconftest -o addopts='' \
  tests/benchmarks/test_qwen3_nightly_contract.py
```

Before enabling the new nightly configuration, qualify both deployments on
CUDA and A3, including the newly enabled A3 Thinker prefix cache. Verify the
full dataset counts, quality floors, usable audio metrics and saved artifacts.
Only then retire the old nightly performance executions. Performance references
remain in collection mode until the repeated target-hardware measurements are
available.
