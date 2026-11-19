---
name: diffusion-perf-test
description: Add or extend nightly/weekly performance-benchmark CI coverage for an existing vLLM-Omni diffusion image or video model, using the unified run_benchmark.py config schema, Buildkite Perf steps, and retained result artifacts. Use when adding performance watch cases and their CI wiring; numerical accuracy testing is diffusion-accuracy-test and optimization diagnosis is diffusion-perf-opt.
---

# Add diffusion performance tests

Add reproducible performance-benchmark coverage for a working diffusion model
or acceleration feature, using the existing perf JSON schema, runners, and
Buildkite conventions. Default to one representative real-weight benchmark per
distinct deployment row. Prefer a workload that exercises more of the model's
input and output paths and the deployment you intend to watch, rather than an
arbitrary resolution or step count.

This workflow covers the **CI coverage** half of performance work: benchmark
cases, their measurement contract, and the nightly/weekly wiring. It does not
diagnose or implement optimizations. For that, use
[diffusion-perf-opt](../diffusion-perf-opt/SKILL.md); for the numerical quality
gate that every optimization and benchmark change must still satisfy, use
[diffusion-accuracy-test](../diffusion-accuracy-test/SKILL.md).

For a model that is not yet integrated, first use
[add-diffusion-model](../add-diffusion-model/SKILL.md) to establish a working
runtime, then return here. An already-serving model can use this skill
directly.

## Workflow

1. **Choose extension or first case.** For an existing case, locate its config
   object, runner, CI step, and source-dependency alias; reuse them within their
   validated scope and inspect only the requested change. For a first case,
   inspect the model's registry/pipeline, supported tasks/endpoints, and a
   sibling model's config, then pick a reference config from
   [config-schema.md](references/config-schema.md#choose-the-closest-existing-case).
   Record the covered deployment rows (device, cards, parallelism) and any
   remaining gap. More prompts or steps alone do not add a coverage row.

2. **Confirm the schema and runner before writing JSON.** Read
   [config-schema.md](references/config-schema.md). Confirm whether the model's
   cases belong to `run_benchmark.py` (omni-bench schema: `dataset_name`) or the
   legacy `run_diffusion_benchmark.py` (`dataset`); do not mix them in one file.
   A new image/video generation case uses `run_benchmark.py`. Resolve the
   runner/schema decision before editing CI so
   [the matching-runner check](references/ci-wiring.md#validate-collection-routing-and-execution)
   passes.

3. **Implement the benchmark case.** Add a `test_name` object to
   `tests/dfx/perf/tests/test_<model>_vllm_omni.json` with `server_params`
   (model + `serve_args`) and one or more `benchmark_params`. Prefer server
   reuse across cases, and give every benchmark entry a stable `name`. Follow
   [workloads.md](references/workloads.md) to choose image vs video parameters
   and the parallelism row.

4. **Fix the measurement contract.** One benchmark row answers one question on
   one deployment. Keep the profiler off for baseline numbers, warm up, record
   the device and parallelism actually used, and report durations in ms. Follow
   [measurement-contract.md](references/measurement-contract.md) so the saved
   result JSON is comparable across runs.

5. **Wire execution into CI.** Read [ci-wiring.md](references/ci-wiring.md).
   Match the runner, `BENCHMARK_DIR`, marks, hardware preset, artifact paths,
   and source-dependency alias in the correct nightly (or weekly) X2I/X2V Perf
   step. A `mark` alone does not make Buildkite execute the test.

6. **Validate, diagnose, and stop.** Collect the intended nodes with the exact
   CI selector and confirm the node ids are exactly the intended rows
   (`pytest --collect-only -q`). Before a long sweep, run one small row from a
   temporary config (for example the 512x512 row with `num_prompts: 1`) to
   confirm the server starts, the model and tokenizer resolve, the attention
   backend is selected, and a real (non-template) result is written; then run
   the full set. The runner's config-load line is conditional: with a
   unified-only `--test-config-file` it prints nothing, and the
   `--test-config-file: loaded N ...; skipped M ...` line appears only when
   `M > 0` legacy-schema cases were filtered out — `skipped` there is not a
   pytest skip count. Each sweep step must produce a **real** benchmark result,
   not the `result_omni_template.json` fallback the runner writes when no result
   file was generated. On failure, distinguish a config mistake (empty sweep,
   missing `tokenizer`, wrong runner) from a real regression before re-running.
   Use the
   [completion and rerun rules](references/ci-wiring.md#completion-and-rerun-scope).
   Report passed, failed, skipped, and unvalidated rows separately. Without
   suitable hardware, deliver feasible edits and exact commands with execution
   explicitly unvalidated; collection is not a pass.

7. **Hand off baselines and report state.** Follow
   [baseline-and-report.md](references/baseline-and-report.md). Note whether the
   row is a pure watch case (no threshold) or carries a `baseline` block, and
   whether any automatic regression gate exists. Do not present a benchmark that
   only ran once as a stable baseline.

Read [vllm-omni-test](../vllm-omni-test/SKILL.md) and its
[test routing reference](../vllm-omni-test/references/test-routing.md) when
fixture, marker, or generic CI mechanics need more detail. Where those describe
the legacy diffusion runner for an endpoint that now uses `run_benchmark.py`,
prefer the concrete runner/schema described in this skill's references.

## Deliver when invoked for a model

- The minimal perf JSON case and CI edits requested, using the correct runner
  and schema.
- The chosen deployment rows (device, cards, parallelism), the input/output paths
  each exercises, and which supported rows remain unvalidated.
- The measurement contract actually used (backend pinning, warmup, profiler
  state, generator/RNG handling) and where the result JSON files are written.
- Copy-paste local single-case and CI-equivalent commands, with `BENCHMARK_DIR`,
  required env vars, hardware, and checkpoint cache/access prerequisites. On an
  offline or uncached machine, set `HF_HUB_OFFLINE=1` so a missing checkpoint
  fails fast instead of retrying.
- For a row with a `baseline` block: the observed values, the hardware bucket the
  comparison uses, and whether that comparison is enforced. State clearly if no
  automatic gate exists.
- Execution results and the precise scope of the performance claim; separate
  measured rows from unvalidated ones.
