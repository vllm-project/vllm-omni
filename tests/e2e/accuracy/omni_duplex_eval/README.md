# Omni-DuplexEval CI Guard

Nightly end-to-end scoring regression for [Hothan/Omni-DuplexEval]
(RTD 6 subtasks + PR 3 subtasks), driven by `vllm bench omni-duplex-eval`
(generate → evaluate → summarize).

## Files

| File | Purpose |
| ------ | --------- |
| `test_omni_duplex_eval_ci.py` | Main pytest: hardcoded thresholds + 6 assertions |
| `test_omni_duplex_eval_ci_mock.py` | CPU mock integration tests (no GPU / model needed) |
| `omni_duplex_eval_ci_config.json` | Dataset / revision / sample IDs / judge |
| `conftest.py` | `judge_server` fixture (endpoint resolution only) |
| `README.md` | This document |
| `results/.gitignore` | Track empty results dir for Buildkite artifact upload |

## Run locally

Judge server must be pre-started; this test never launches it:

```bash
# CUDA (H100 / B200), or NPU (A3)
VLLM_DUPLEX_EVAL_JUDGE_URL=http://127.0.0.1:8001 \
  pytest -s -v tests/e2e/accuracy/omni_duplex_eval/ \
    -m "full_model and cuda and (H100 or B200) and cards_1"
```

The `omni_server` fixture boots MiniCPM-o-4_5 with the community duplex
params (`tests/e2e/online_serving/helpers/minicpmo_4_5_duplex.py`).
`judge.base_url_env` (default `VLLM_DUPLEX_EVAL_JUDGE_URL`) overrides
`judge.base_url` from the JSON config.

## Results output

After Phase 3 (summarize), the full `summarize_scores()` dict is persisted
to `results/summary.json` (same pattern as PR#6817 perf tests) and printed
to stdout:

- `results/` is tracked in git via `results/.gitignore` (empty dir, nothing
  committed inside).
- On Buildkite nightly, the step uploads `results/*.json` as artifacts
  (`buildkite-agent artifact upload`) even when assertions fail, so a
  failing run's scores remain downloadable.

## Judge server

Judge is a pre-started, OpenAI-compatible endpoint (never launched by this
test). It can run on a separate machine / GPU from the `omni_server`.

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
`vllm bench omni-duplex-eval evaluate` for materialized videos.

## CPU mock tests (no GPU / model / judge needed)

`test_omni_duplex_eval_ci_mock.py` drives the *real*
`test_omni_duplex_eval_ci()` guard on CPU by stubbing the heavy I/O:

- `_run_cli` is replaced by an argv-recording spy (never executes).
- `resolve_ref_audio` is stubbed to a local path.
- Fake score files are pre-written under `tmp_path/scores/<split>/` from
  `sample_ids_per_split` (minus `exclude_ids`), so `summarize_scores()`
  and the Phase-4 assertions run for real.

It validates the CLI argument assembly (generate + evaluate), the
score-file-count check, `protocol_pin`, the RTD/PR `by_task` keys, and
all threshold comparisons — including the failure paths. Runs in CI on
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
produces real scores and persists `results/summary.json` without failing the
gate. Tighten after the first nightly run:

1. Record `summarize_scores()` output (mean content / temporal / success).
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

1. `protocol_pin == PROTOCOL_PIN` (track upstream protocol changes)
2. `summary["samples"]` == number of selected IDs
3. RTD `by_task` keys == 6 RTD subtasks
4. PR `by_task` keys == 3 PR subtasks
5. RTD mean content / temporal scores >= thresholds
6. PR `mean_all_success` >= threshold

## Reference paper & scoring scale

Reference paper: **Omni-DuplexEval** (arXiv:2605.17360). The paper reports
a 100-point scale, while the evaluation code
(`vllm_omni/benchmarks/duplex/omni_duplex_eval_metrics.py`) uses a 3-point
content scale (0.00–3.00), a 0–3 integer temporal scale, and a 0/1 PR
success flag. The CI thresholds are defined against the **code's** scales
(the source of truth for `summarize_scores()` output), not the paper's
100-point scale.
