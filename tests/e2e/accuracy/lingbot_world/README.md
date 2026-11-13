# LingBot World accuracy

The CUDA profile (`configs/cuda.json`) carries WBench and VBench `score_floors`
from vllm-omni Buildkite build 19532 at
`d60e82c0d5deeaed6c7bbcb7725f12d02e694f82`: each recorded score minus 0.05,
rounded down to four decimals. The largest per-dimension difference against
CUDA build 18935 was 0.0175. Binary `dynamic_degree` is recorded only. Without
`--lingbot-record-only`, those suites report `comparison_status: compared` and
fail when any listed score is below its floor.

Numerical precision has no stored reference tensors, so tensor error, PSNR and
SSIM remain `pending_baseline`; the test checks only that the fixed-input replay
is complete and finite. Profiles without floors (NPU) must run with
`--lingbot-record-only`, which records outputs and scores with
`comparison_status: pending_baseline`.

The three independent tests use the packaged `vllm bench serve --omni` scoring
path after the candidate server has stopped:

- `test_precision.py::test_numeric_precision` records fixed-input Wan VAE NPZ
  arrays, native/stored dtype, shape, finite checks and descriptive statistics.
  Tensor errors, PSNR and SSIM remain pending until genuine reference data exist.
- `test_wbench.py::test_wbench_quality` records native WBench dimensions.
- `test_vbench.py::test_vbench_quality` records native VBench dimensions,
  including full-frame background inference coverage.

The session fixture starts the candidate server once and captures one 40-chunk
rollout for each of the three existing image/prompt/camera cases. Each case has
an excluded eight-chunk warmup. It runs one fixed-input VAE replay separately.
There is no official source checkout, live reference server, or repeated
reference/candidate campaign in routine CI.

Each test checks its own authenticated inputs, prints real measurements as JSON,
and writes `result_acc_*.json` before asserting recording validity. A missing
NPZ fails numerical recording without hiding complete videos from the semantic
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
python -m tests.e2e.accuracy.lingbot_world.prepare capture --record-only \
  --profile tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --output lingbot-artifacts/npu-accuracy
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-record-only \
  --lingbot-profile=tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --lingbot-generation=lingbot-artifacts/npu-accuracy \
  --lingbot-output=lingbot-artifacts/npu-accuracy-results
```

CI saves the numerical NPZ, videos, raw evaluator scores, workload/source
manifest and JSON reports as artifacts and prints measurements with `pytest -sv`.
NPU score floors and precision reference data need their own recorded runs.
Score floors come only from reviewed CI measurements; new results are never
auto-promoted to a baseline.

Partial capture preserves its manifest and available artifacts. The three
tests inspect their own required inputs even after a producer failure. A local
run without `--lingbot-profile` is opt-in; selected CI tests cannot skip.
Each accuracy job has a 120-minute timeout budget; actual native duration and
two-GPU capacity still need matching CI evidence.

For a CPU-only check of capture/recording contracts:

```bash
python -m pytest -q tests/benchmarks/test_lingbot_world_cases.py \
  tests/benchmarks/test_lingbot_world_precision.py \
  -m "core_model and cpu" --run-level core_model
```

Performance remains in `tests/dfx/perf` with its recorded JSON workflow.
There is no stability or weekly LingBot test lane.
