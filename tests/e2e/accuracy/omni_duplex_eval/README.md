# Omni-DuplexEval CI Guard

Nightly end-to-end scoring regression for [Hothan/Omni-DuplexEval]
(RTD 6 subtasks + PR 3 subtasks), driven by `vllm bench serve --omni`
with integrated generation, judging, and result saving.

## Files

| File | Purpose |
| ------ | --------- |
| `test_omni_duplex_eval_ci.py` | Main pytest: benchmark invocation, coverage checks, and score thresholds |
| `test_omni_duplex_eval_ci_mock.py` | CPU mock integration tests (no GPU / model needed) |
| `omni_duplex_eval_ci_config.json` | Dataset / revision / sample IDs / judge |
| `conftest.py` | `judge_server` fixture and model/judge device isolation |
| `README.md` | This document |
| `results/.gitignore` | Track empty results dir for Buildkite artifact upload |

## Run locally

The test needs two devices: one for MiniCPM-o and one for the judge.
It starts the judge unless `VLLM_DUPLEX_EVAL_JUDGE_URL` points to a healthy server:

```bash
# CUDA (H100 / B200); for NPU use "full_model and npu and A3 and cards_2".
pytest -s -v tests/e2e/accuracy/omni_duplex_eval/test_omni_duplex_eval_ci.py \
  -m "full_model and cuda and (H100 or B200) and cards_2" \
  --run-level full_model --basetemp /tmp/omni_duplex_ci
```

The `omni_server` fixture boots MiniCPM-o-4_5 with the community duplex
params (`tests/e2e/online_serving/helpers/minicpmo_4_5_duplex.py`).
The benchmark selects every configured sample ID with `--num-prompts 0`,
`--disable-shuffle`, and `--max-concurrency 1`. Warmups and the benchmark's
request-based readiness probe are disabled; the server fixture already waits
for startup. Input replay uses the default real-time media clock and 1 FPS.
The judge settings continue to come from the JSON config and `judge_server`.

## Results output

`bench serve --duplex-eval-evaluate --save-result` prints the performance and
accuracy results and writes:

- `results/bench-result.json`: full serving benchmark result, including
  `omni_duplex_eval.accuracy`, publication errors, and
  `published_with_unfinished_responses`.
- `<response-root>/run-*/evaluation/evaluation_summary.json`: the accuracy
  summary saved by the benchmark.

The guard copies `bench-result.json`'s `omni_duplex_eval.accuracy` to
`results/result_acc_{model}_{dataset}_{date}_{random_number}.json` for
dashboard collection. Model and dataset use the basenames of their configured
IDs or paths. The date is UTC in `YYYYMMDD-HHMMSS` format, followed by an
eight-digit random number, for example:
`result_acc_MiniCPM-o-4_5_Omni-DuplexEval_20261014-070221_48392107.json`.
It does not run a second evaluation or summarization. Before each run, the
guard removes the previous `bench-result.json` and `result_acc_*.json` files
from its results directory so stale results cannot be uploaded as current
results after an early CLI failure. The accuracy summary is saved before
checking the results:

- `results/` is tracked in git via `results/.gitignore` (empty dir, nothing
  committed inside).
- On Buildkite nightly, the step uploads `results/*.json` as artifacts
  (`buildkite-agent artifact upload`) even when assertions fail, so a
  failing run's scores remain downloadable.

## Judge server

The fixture starts an OpenAI-compatible judge on the first allocated device
and the Omni server on the second. A healthy pre-started judge can be reused
through `VLLM_DUPLEX_EVAL_JUDGE_URL`.

Current judge: **Qwen2.5-VL-7B-Instruct** (`judge.model` in the JSON
config). The 7B model is chosen for Phase A to run on a single GPU (≈16 GB
FP16, or ≈5 GB with INT4 quantization); PR 0/1 classification and temporal
0–3 scoring are robust at this size. The 0.00–3.00 content scoring is the
weakest link (fine-grained counting / color / spatial checks) — if the
guard proves insensitive to regressions, upgrade the judge (e.g.
72B on FP8 H100 or an external API) and **re-run the baseline** to reset
`_MIN_*` thresholds. Changing the judge never affects the P0 assertions
(`protocol_pin`, sample count, `by_task` keys) or the CPU mock tests.

Example (any OpenAI-compatible vLLM server):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --max-model-len 65536 \
  --allowed-local-media-path /tmp/omni_duplex_ci
```

If the judge is started on a different host or the media temporary directory
changes, adjust `--allowed-local-media-path` to match the path used by
`bench serve` for materialized videos. A remote judge must be able to read
those paths when using the default `video_url` mode.

## CPU mock tests (no GPU / model / judge needed)

`test_omni_duplex_eval_ci_mock.py` drives the *real*
`test_omni_duplex_eval_ci()` guard on CPU by stubbing the heavy I/O:

- `_run_cli` is replaced by an argv-recording spy that saves a synthetic
  benchmark result (no model execution).
- `resolve_ref_audio` is stubbed to a local path.
- Fake score files are pre-written under a run's `evaluation/<split>/`
  directory from `sample_ids_per_split` (minus `exclude_ids`), so the
  summarizer and guard assertions run for real.

The tests parse the assembled command with the real `bench serve` parser,
verify artifact persistence, and reject incomplete generation, publication,
or judging even when the CLI exits successfully. They retain coverage of
score files, protocol pins, and score thresholds. Runs in CI on
every PR (`core_model` + `cpu` marks):

```bash
pytest -v tests/e2e/accuracy/omni_duplex_eval/test_omni_duplex_eval_ci_mock.py
```

## Thresholds (record mode → tighten after first nightly run)

`sample_ids_per_split` contains real dataset IDs (verified against the
Hothan/Omni-DuplexEval splits; PR_correction / PR_event_reminder IDs are
inferred from the split row order and must be re-verified once the dataset
is reachable). `exclude_ids` has been removed.

All `_MIN_*` thresholds are currently **0.0 (record mode)**: the nightly run
produces real scores and persists `results/result_acc_*.json` without failing the
gate. Tighten after the first nightly run:

1. Record the run's `results/result_acc_*.json` (mean content / temporal / success).
2. Set `_MIN_*` thresholds = `mean - n * std` (start `n=2`) in
   `test_omni_duplex_eval_ci.py`. As a starting point, the paper's
   MiniCPM-o 4.5 scores (100-point scale) map to ≈1.15 content / ≈2.40
   temporal / ≈0.20 PR on the code scales (see "Reference paper" below);
   ~50% of those is a reasonable lower bound.
3. If needed, exclude known bad samples (generate hang / judge failure) via
   `exclude_ids`.

Thresholds are bound to the judge model: **switching the judge requires
re-running steps 1–2**. If the 7B judge shows high variance (small models
are more prompt-sensitive), widen `n` (e.g. `n=3`) or increase the sample
size before tightening.

## Assertions (in order)

1. All selected requests completed and were published, with no request or artifact errors.
2. Evaluation completed for every selected sample, with no failed/skipped evaluations or errors.
3. Config, individual scores, and summary match `PROTOCOL_PIN`.
4. Score file count and split/ID set match the CI selection; summary sample count matches too.
5. RTD `by_task` covers all 6 RTD subtasks, and PR covers all 3 PR subtasks.
6. RTD mean content / temporal scores and PR `mean_all_success` meet the thresholds.

Samples with unfinished responses remain eligible for scoring, following the
benchmark's policy. Their count is recorded in `bench-result.json`; it is
not an additional pass/fail threshold.

## Reference paper & scoring scale

Reference paper: **Omni-DuplexEval** (arXiv:2605.17360). The paper reports
a 100-point scale, while the evaluation code
(`vllm_omni/benchmarks/duplex/omni_duplex_eval_metrics.py`) uses a 3-point
content scale (0.00–3.00), a 0–3 integer temporal scale, and a 0/1 PR
success flag. The CI thresholds are defined against the **code's** scales
(the source of truth for `summarize_scores()` output), not the paper's
100-point scale.
