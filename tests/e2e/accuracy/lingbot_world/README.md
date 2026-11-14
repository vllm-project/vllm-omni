# LingBot World accuracy

Each profile carries WBench and VBench `score_floors`: every recorded score
minus 0.05, rounded down to four decimals. Binary `dynamic_degree` is reported
but has no floor. The semantic suites report `comparison_status: score_floor`
and fail when any listed score is missing or below its floor. A profile without
floors is rejected.

- CUDA (`configs/cuda.json`): vllm-omni build 19532 at
  `d60e82c0d5deeaed6c7bbcb7725f12d02e694f82`. The largest per-dimension
  difference against CUDA build 18935 was 0.0175.
- NPU (`configs/npu.json`): vllm-omni-npu-ci build 11412 at
  `49a7059849dd235e26ce43221bf33c06ea2f7ba7` (A2 generation, H100 scoring).
  The largest per-dimension difference against CUDA build 19532 was 0.0499.

Known limitation: numerical precision has no stored reference tensors, so
tensor error, PSNR and SSIM are not computed. `test_numeric_precision` reports
`comparison_status: finite_only` and checks only that the fixed-input replay is
complete and finite. Adding a reference needs an immutable artifact location
for the NPZ; it is not generated from the candidate.

The three independent tests use the packaged `vllm bench serve --omni` scoring
path after the candidate server has stopped:

- `test_precision.py::test_numeric_precision` reports fixed-input Wan VAE NPZ
  arrays, native/stored dtype, shape, finite checks and descriptive statistics.
- `test_wbench.py::test_wbench_quality` checks native WBench dimensions against
  the profile floors.
- `test_vbench.py::test_vbench_quality` checks native VBench dimensions,
  including full-frame background inference coverage, against the profile
  floors.

The session fixture starts the candidate server once and captures one 40-chunk
rollout for each of the three existing image/prompt/camera cases. Each case has
an excluded eight-chunk warmup. It runs one fixed-input VAE replay separately.
There is no official source checkout, live reference server, or repeated
reference/candidate campaign in routine CI.

Each test checks its own authenticated inputs, prints real measurements as JSON,
and writes `result_acc_*.json` (with the floors under `baseline`) before
asserting. A missing NPZ fails the numerical test without hiding complete
videos from the semantic
tests. Server/cleanup failures remain failures. Generation hardware, source,
model and workload are separate from the CUDA evaluator hardware.

CUDA generation uses two H100 GPUs with Ulysses/SP=2. NPU uses the existing
four-device A2 configuration, followed by the same scorers on a CUDA worker.
Evaluator dependencies, source revisions and weight checksums remain pinned.

CUDA check (requires two H100 GPUs, pinned model assets and evaluator cache
or network access for initial provisioning):

```bash
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-profile=tests/e2e/accuracy/lingbot_world/configs/cuda.json \
  --lingbot-output=lingbot-artifacts/cuda-accuracy
```

NPU capture requires four A2 devices. Transfer the whole capture directory to
the CUDA evaluator worker; consuming it does not prepare or require an NPU
model runtime on that worker:

```bash
python -m tests.e2e.accuracy.lingbot_world.prepare capture \
  --profile tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --output lingbot-artifacts/npu-accuracy
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-profile=tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --lingbot-generation=lingbot-artifacts/npu-accuracy \
  --lingbot-output=lingbot-artifacts/npu-accuracy-results
```

CI saves the numerical NPZ, videos, raw evaluator scores, workload/source
manifest and JSON reports as artifacts and prints measurements with `pytest -sv`.
Score floors come only from reviewed CI measurements; new results are never
auto-promoted to a baseline.

Partial capture preserves its manifest and available artifacts. The three
tests inspect their own required inputs even after a producer failure. A local
run without `--lingbot-profile` is opt-in; selected CI tests cannot skip.
The NPU profile allows 2400 s for server startup: serial shard loading on a
contended A2 host took 57-126 s per shard (vllm-omni-npu-ci build 11393), versus
~22 s on earlier runs. Job timeouts follow measured nightly durations:
CUDA accuracy 60 min (21.7-25.2 min, builds 19532-19586), NPU Data 90 min
(22.2-22.9 min; the 2400 s startup bound plus ~15 min of capture fits), NPU
Outputs 45 min (12.3-15.0 min). Perf jobs: CUDA 45 min (6.9-8.1 min), A2
60 min (15.9-19.2 min; server init is bounded at 900 s).

For a CPU-only check of capture and scoring contracts:

```bash
python -m pytest -q tests/benchmarks/test_lingbot_world_cases.py \
  tests/benchmarks/test_lingbot_world_precision.py \
  -m "core_model and cpu" --run-level core_model
```

Performance remains in `tests/dfx/perf` with its recorded JSON workflow.
These accuracy and performance workflows add no stability or weekly lane.
Existing DFX stability cases keep
`benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py` and their deploy
config paths. This compatibility entry delegates to the shared serving benchmark.
