# CI wiring

A perf JSON file is inert until a Buildkite step runs it with the matching
runner on the right pool and uploads the results. Wire all of it, then confirm
the step actually collected and executed.

## Files and responsibilities

| File | Change |
|---|---|
| `tests/dfx/perf/tests/test_<model>_vllm_omni.json` | The benchmark case(s) |
| `.buildkite/cuda/test-nightly.yml` | CUDA nightly Perf step in the Diffusion X2I or X2V group |
| `.buildkite/npu/test-npu-nightly.yml` | NPU nightly Perf step (if NPU support is claimed) |
| `.buildkite/cuda/test-weekly.yml` | Only if the row is a `slow`/weekly job (no diffusion Perf step exists here yet) |
| `.buildkite/common/ci_source_file_dependencies.yml` | `diffusion_<slug>_perf` alias (source files that should trigger the step) |
| `.claude/skills/readme.md` | Only when adding this skill, not per model |

Do not add a model-specific helper under `tests/helpers/`; keep constants in the
perf JSON.

## Choose nightly or weekly

- **Nightly** (`test-nightly.yml`, Diffusion X2I / X2V groups): default for a
  performance watch case on real weights.
- **Weekly** (`test-weekly.yml`, `Perf Test` group): heavier or `slow` rows.
  There is currently no diffusion Perf step in the weekly pipeline, so adding one
  is a new wiring decision, not a copy of an existing diffusion row.

Mark the weekly variant with `slow` in the JSON `mark` array, following the omni
`test_qwen3_omni_async_chunk.json` pattern.

## NPU routing

NPU steps live in `test-npu-nightly.yml`, run under `mirror_hardwares`
(`a3_npu_2`, `a3_npu_4`, …), set `VLLM_WORKER_MULTIPROC_METHOD: spawn`, and wrap
the pytest call so artifacts upload even on failure:

```yaml
- label: ":full_moon: Diffusion X2I(&A&T) · Perf Test · <Model> · Single-NPU"
  source_file_dependencies: diffusion_<slug>_perf
  key: nightly-diffusion-x2iat-performance-<slug>-single-npu
  timeout_in_minutes: 180
  mirror_hardwares: a3_npu_2
  env:
    VLLM_WORKER_MULTIPROC_METHOD: spawn
    HF_TOKEN: "${HF_TOKEN}"
    DIFFUSION_ATTENTION_BACKEND: TORCH_SDPA
  commands:
    - export BENCHMARK_DIR=tests/dfx/perf/results
    - |
      set +e
      pytest -s -v tests/dfx/perf/scripts/run_benchmark.py -m "npu and A3 and cards_1" --test-config-file tests/dfx/perf/tests/test_<model>_vllm_omni.json
      EXIT=$$?
      buildkite-agent artifact upload "tests/dfx/perf/results/*.json"
      exit $$EXIT
```

`$$?` is the Buildkite-escaped `$?`. The CUDA step sticks with the normal
`pytest` command and a plain `artifact_paths` list.

## Source-dependency alias

Add a `<model>_perf` alias next to the other `*_perf` entries. Reuse the model's
own source anchor and add the exact runner + config file:

```yaml
diffusion_<slug>_perf:
  - *<model_source_anchor>
  - tests/dfx/perf/scripts/run_benchmark.py
  - tests/dfx/perf/tests/test_<model>_vllm_omni.json
```

Without this, changing the model, runner, or config will not trigger the step.
`source_filter_fallback` lists only
`.buildkite/common/ci_mirror_hardwares.yml` and `tests/helpers/`; a diff limited
to those paths keeps every job in the upload, so leave them out of the alias.
`tests/dfx/` is **not** a fallback prefix: a shared DFX change such as
`tests/dfx/conftest.py` or `tests/dfx/perf/tests/test_runner_metadata.py` matches
no alias and does not select these steps — add such a path to the alias (or
trigger the step explicitly) when the change depends on it.

## CUDA nightly step template

```yaml
- label: ":full_moon: Diffusion X2I(&A&T) · Perf Test · <Model> · Single-GPU"
  source_file_dependencies: diffusion_<slug>_perf
  key: nightly-diffusion-x2iat-performance-<slug>-single
  timeout_in_minutes: 180
  # Pin the backend only when auto-selection cannot run the row (for example a
  # mask the selected kernel rejects); otherwise let the runner auto-select.
  env:
    DIFFUSION_ATTENTION_BACKEND: TORCH_SDPA
  artifact_paths:
    - tests/dfx/perf/results/*.json
  commands:
    - export BENCHMARK_DIR=tests/dfx/perf/results
    - pytest -s -v tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_<model>_vllm_omni.json -m "H100 and B200 and cards_1"
```

Add a parallel `-4gpu` step for the multi-card row, using `-m "H100 and B200 and
cards_4"` and the same alias. Reference: the real Qwen-Image-2.1 steps in
[test-nightly.yml](../../../../.buildkite/cuda/test-nightly.yml).

## New steps are artifact-only

`:email: Nightly Collection & Email` (`key: nightly-perf-distribution`) in
`test-nightly.yml` is deprecated. Do **not** add it to your step's `depends_on`
or otherwise extend it.

Aggregation is a hand-written enumeration, not a glob: that step lists every
producing key in its own `depends_on` and repeats each one as a
`buildkite-agent artifact download ... --step <key>` command, and it runs only
when the build sets `EMAIL_DISTRIBUTION=1`. A new step therefore uploads
`tests/dfx/perf/results/*.json` to the build and stops there. Treat every row
you add as **artifact-only**: its numbers do not reach the nightly
Excel/HTML/email chain, so do not claim they do. Including a row in those
reports means adding its step key to that deprecated step's `depends_on` and
download list, which is a CI-owner decision, not part of adding a row. See
[baseline-and-report.md](baseline-and-report.md).

## Validate collection, routing, and execution

Three separate checks, in order:

1. **Collection.** With the exact CI selector, collect only:
   `pytest --collect-only -q tests/dfx/perf/scripts/run_benchmark.py
   --test-config-file tests/dfx/perf/tests/test_<model>_vllm_omni.json -m "<selector>"`.
   Confirm the expected node ids (`{test_name}-{name}`) appear and none are
   filtered out.
2. **Routing.** Confirm the step renders with the right runner, hardware preset,
   artifact paths, and alias. The uploader tests cover this:
   `tests/buildkite/test_upload_pipeline.py`. The runner/schema match is enforced
   by `tests/dfx/perf/tests/test_runner_metadata.py::test_buildkite_perf_steps_use_matching_runner_schema`
   — run it after editing pipelines (a mismatch makes pytest select 0 tests and
   the step exit 5, issue #8074).
3. **Execution.** Run on suitable hardware. Start with one small row from a
   temporary config (for example the 512x512 row with `num_prompts: 1`) to
   confirm server start, model/tokenizer resolution, backend selection, and a
   real result before spending the step timeout on the full set. The runner's
   config-load line is conditional: with a unified-only `--test-config-file` it
   prints nothing, and `--test-config-file: loaded N ...; skipped M ...` appears
   only when `M > 0` legacy-schema cases were filtered out. Every sweep step must
   produce a **real** benchmark result under `BENCHMARK_DIR`. An empty sweep
   (wrong key case) indicates a config bug, not a passing benchmark. File
   existence alone is not proof: when the benchmark generates no result,
   `run_benchmark()` writes the `result_omni_template.json` fallback to the
   result path and prints `Benchmark result file not generated, fallback to
   template`, so inspect the JSON contents (`test_name`, `name`, `model_id`,
   `completed`, `baseline`) and the console log before calling the row a pass.

## Completion and rerun scope

Report per row: passed / failed / skipped / unvalidated, using the actual
selector and hardware. Re-run only the rows affected by a fix — do not re-run
the whole file to paper over one failing row. Without access to the required
hardware, deliver the feasible edits and the exact commands, mark execution
explicitly unvalidated, and do not call collection a pass.

`BENCHMARK_DIR` is passed to `vllm bench serve` as `--result-dir` and falls back
to `tests` when unset; every perf CI step points its result directory at
`tests/dfx/perf/results` (already gitignored) — unified-runner steps export
`BENCHMARK_DIR`, legacy diffusion steps export `DIFFUSION_BENCHMARK_DIR`. The
directory is not cleaned between runs and filenames carry a timestamp. When
comparing or uploading, filter by the current run's timestamp (or clear the
directory first) so a stale file is not read as the current result.
