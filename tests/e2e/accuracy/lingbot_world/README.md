# LingBot World accuracy recording

The temporary `--lingbot-record-only` mode records authentic candidate outputs
and scores for a follow-up baseline PR. A successful test means capture/scoring
completed with valid data. Every report states `comparison_status:
pending_baseline`; it does not establish that accuracy has not regressed.

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

CUDA recording (requires two H100 GPUs, pinned model assets and evaluator cache
or network access for initial provisioning):

```bash
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-record-only \
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
After those jobs run, a follow-up PR can review the data and add real baseline
values and immutable artifact references. This PR does not generate invented
score floors, compare a video to itself, or auto-promote new results to a
regression baseline.

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
