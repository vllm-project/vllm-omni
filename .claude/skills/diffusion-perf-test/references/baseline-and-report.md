# Baselines, regression gating, and reporting

State clearly, per row, whether it is a pure watch case or carries a baseline,
and whether anything actually enforces that baseline.

## The `baseline` block

`baseline` is an optional key on a `benchmark_params` entry. It is
**hardware-nested**: top-level keys must be hardware pytest markers from
`pyproject.toml`, and metric names/values are free-form (scalars or
sweep-aligned lists):

```json
"baseline": {
  "H100": {"throughput_qps": 1.2, "latency_mean": 950.0, "peak_memory_mb_mean": 33372.0},
  "A3":   {"throughput_qps": [1.0, 1.1]}
}
```

`resolve_baseline_for_sweep()` (in [tests/dfx/conftest.py](../../../../tests/dfx/conftest.py))
reduces each metric to the value for this sweep step; a non-hardware-nested
baseline raises. Keep list lengths aligned with the sweep.

Use the result field names of the runner that would read them. The example above
uses the legacy runner's names (`throughput_qps`, `latency_mean`,
`peak_memory_mb_mean`); a unified-runner row should use `request_throughput`,
`mean_e2el_ms`, and `mean_peak_memory_mb`. `mean_tpot_ms` is the exception: it
is the only key that has an effect today (`assert_result` checks its finiteness;
see below). Nothing else consumes these names yet, but a future gate should use
the runner's own field names.

The runtime `Hardware` value selects the bucket. If it is not one of the keys
(for example a machine whose device name normalizes to an unlisted token), the
baseline is recorded but never matched.

## What a baseline actually does today

With the unified runner, `run_benchmark()` writes the resolved `baseline` into
the result JSON (`result["baseline"]`). `assert_result` only asserts metric
finiteness when a *TPOT* baseline is configured
(`baseline[<hardware>]["mean_tpot_ms"]`); it does **not** compare throughput or
latency against the configured thresholds. So:

- A `baseline` block is **recorded evidence**, not an enforced gate.
- Omitting `baseline` runs the case without any threshold assertion.
- `--assert-baseline` is registered in `tests/dfx/conftest.py`
  (`pytest_addoption`) with help text saying "default: off", but **no runner
  consumes it** — grep for `getoption` / `assert-baseline` in the runners returns
  nothing.

Do not tell a user a regression is "gated" when the value is only recorded.

## Proposing an enforced regression gate (new capability)

If the task is to make regressions fail CI, that is new work, not wiring:

1. Decide the comparison: last-known-good committed baseline, or previous run.
2. Decide the metric and tolerance per row, and the hardware bucket.
3. Implement the consumer (extend the runner to read `--assert-baseline`, or add
   a compare script) and a place to store baselines.
4. Re-run the rows enough times to set a defensible tolerance; quote the spread.

Keep this separate from "add a watch case", which only records results.

## Nightly report chain

A directory of results (`tests/dfx/perf/results/*.json`, i.e. `BENCHMARK_DIR`)
is the input to the nightly reporting tools:

- `tools/nightly/generate_nightly_perf_excel.py` (sheets: `omni_summary`,
  `diffusion_summary`, `omni_raw`, `diffusion_raw`)
- `tools/nightly/generate_nightly_perf_html.py`
- `tools/nightly/send_nightly_email.py`

The `:email: Nightly Collection & Email` step (`key: nightly-perf-distribution`)
in `test-nightly.yml` runs them, but it is **deprecated** and collects by hand:
it downloads results only from the step keys hard-coded in its own `depends_on`.
A row added to a new step therefore stays **artifact-only** — uploaded to the
build, absent from these reports. See
[ci-wiring.md](ci-wiring.md#new-steps-are-artifact-only).

## Reporting a row

For each measured row, report: the deployment (device, cards, parallelism), the
workload (endpoint, resolution, steps), the observed metric with spread, the
hardware bucket a `baseline` comparison would use, and whether that comparison is
enforced. Label profiler-instrumented runs separately from baseline runs, and
keep unvalidated rows visibly separate from measured ones.

For a concurrency or QPS sweep, report every point, not the row as one number:
each point writes its own result file and its `mean_e2el_ms` includes queueing.
Pair throughput with latency (`image_throughput` / `request_throughput` for the
scaling claim) and do not call a high-concurrency E2E increase a regression
against a single-point row. Keep the prompts-per-concurrency ratio consistent
across points when comparing them.
