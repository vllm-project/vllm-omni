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
`mean_e2el_ms`, and `mean_peak_memory_mb`. With `--assert-baseline`, every
finite metric in the matching hardware bucket is compared (see below).
`mean_tpot_ms` is also checked for a finite sample even when the flag is off.

The runtime `Hardware` value selects the bucket. If it is not one of the keys
(for example a machine whose device name normalizes to an unlisted token), the
baseline check fails with an error.

## What a baseline does

With the unified runner, `run_benchmark()` writes the sweep-resolved `baseline`
into the result JSON (`result["baseline"]`). The legacy diffusion runner stores
the same resolved map on the metrics it returns.

`--assert-baseline` (registered in `tests/dfx/conftest.py`, default off) makes
both runners call `assert_baseline_within_tolerance()` in
`tests/dfx/conftest.py`:

- Compare the bucket whose key is the runtime `Hardware` value. Missing or
  unmatched hardware fails the check.
- Throughput and QPS (`throughput`, `qps` in the metric name) fail when the
  measured value is **more than 10% below** the baseline (`measured < baseline * 0.9`).
- Latency, memory, and real-time factor (`latency`, `ttft`, `tpot`, `itl`,
  `e2el`, `ttfp`, `rtf`, `memory`) fail when the measured value is **more than
  10% above** the baseline (`measured > baseline * 1.1`).
- An exact 10% boundary passes. A metric name that matches neither family fails
  the case instead of being skipped.
- Omitting `baseline`, or leaving it empty, still runs the case with no
  threshold assertion. The existing TPOT finiteness check still runs whenever
  `mean_tpot_ms` is configured, even without the flag.

CUDA and NPU perf jobs pass `--assert-baseline`. AMD jobs do not, until that
hardware has its own bucket. A run without the flag records the baseline and
does not apply this gate. Do not call a row gated unless the command includes
the flag and the result hardware matches a bucket.

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
