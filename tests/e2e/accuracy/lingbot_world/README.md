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

The numerical profiles contain recorded `precision_baseline` values from
successful CUDA build [19900](https://buildkite.com/vllm/vllm-omni/builds/19900)
and NPU build [11754](https://buildkite.com/vllm/vllm-omni-npu-ci/builds/11754),
both at `980d9e5858956e1dfdd50be294bf755799986846`. Source, model, hardware,
artifact IDs and NPZ SHA256 hashes are retained beside the values. Normal CI
reads these baselines; it never regenerates them from the candidate.

`test_numeric_precision` checks the existing seed-42, single-device BF16 Wan
VAE decode of a fixed latent. It records 64 C-order points per tensor at
`i * (size - 1) // 63`, including both endpoints, and compares the 64 decoded
output values with that generation platform's baseline. These points cover
every output channel and temporal slice. The full stored input-array SHA256,
operation/seed, tensor shapes, native/stored dtypes and exact sample positions
must match. Small test tensors use `min(64, size)` points, evenly spaced with
the same rule. Missing, empty, nonfinite or incompatible baselines fail.

The initial output tolerance is absolute `0.04`, with zero relative tolerance.
For normalized `[-1, 1]` output this is approximately 5.1 RGB levels before
quantization, informed by Bagel's fixed-pixel regression convention. It is not
calibrated from repeated LingBot runs. CUDA and NPU have separate recorded
values and input hashes; an NPU capture evaluated on H100 uses the NPU
baseline. Source/software revisions are recording provenance and need not
equal a future candidate revision; the pinned model/assets contract must match.

This is a sparse VAE regression check, not official-model equivalence or
denoiser/full-rollout tensor precision. Full-tensor error, PSNR and SSIM are
not computed. The actual generated videos remain independently covered by
WBench and VBench. The numerical result reports
`comparison_status: sampled_baseline`, actual sample values, maximum absolute
sample errors, tolerance violations and the recorded baseline source.

The three independent tests use the packaged `vllm bench serve --omni` scoring
path after the candidate server has stopped:

- `test_precision.py::test_numeric_precision` compares recorded Wan VAE output
  samples and reports native/stored dtype, shape, input identity, finite checks
  and descriptive statistics.
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
`--realtime-video-warmup-num-chunks` sets the separate warmup rollout length;
`--realtime-video-warmup-chunks` excludes the initial attention-window chunks
from steady-cadence statistics in a measured rollout.

Each test checks its own authenticated inputs, prints real measurements as JSON,
and writes `result_acc_*.json` (with the score floors or expected output points
under `baseline`) before
asserting. A missing NPZ fails the numerical test without hiding complete
videos from the semantic
tests. Server/cleanup failures remain failures. Generation hardware, source,
model and workload are separate from the CUDA evaluator hardware.

CUDA capture requires released vLLM 0.34.0 and Torch 2.14.0. NPU capture keeps
its existing vLLM-Ascend 0.33.0 image and is not qualified for vLLM 0.34.0.

CUDA generation uses two H100 GPUs with Ulysses/SP=2. NPU uses the existing
four-device A2 configuration, followed by the same scorers on a CUDA worker.
Evaluator dependencies, source revisions and weight checksums remain pinned.
`requirements/video-quality.in` and `requirements/video-quality.txt` describe
the independent WBench/VBench evaluator environment. They are not model
documentation or default inference dependencies; provisioning syncs the lock
into a separate Python 3.12 virtual environment for the older evaluator APIs.

CUDA check (requires two H100 GPUs, pinned model assets and evaluator cache
or network access for initial provisioning):

```bash
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-profile=tests/e2e/accuracy/lingbot_world/configs/cuda.json \
  --lingbot-output=tests/e2e/accuracy/results/lingbot_world/cuda
```

NPU capture requires four A2 devices. Transfer the whole capture directory to
the CUDA evaluator worker; consuming it does not prepare or require an NPU
model runtime on that worker:

```bash
python -m tests.e2e.accuracy.lingbot_world.prepare capture \
  --profile tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --output tests/e2e/accuracy/results/lingbot_world/npu-capture
python -m pytest -sv --run-level full_model \
  tests/e2e/accuracy/lingbot_world/test_precision.py \
  tests/e2e/accuracy/lingbot_world/test_wbench.py \
  tests/e2e/accuracy/lingbot_world/test_vbench.py \
  --lingbot-profile=tests/e2e/accuracy/lingbot_world/configs/npu.json \
  --lingbot-generation=tests/e2e/accuracy/results/lingbot_world/npu-capture \
  --lingbot-output=tests/e2e/accuracy/results/lingbot_world/npu
```

CI prints measurements with `pytest -sv`. Standard accuracy records use the
shared `write_accuracy_result` destination, `tests/e2e/accuracy/results`,
respecting `ACC_BENCH_RESULT_DIR` when set. CI uploads these result files:

| Result | Path |
| --- | --- |
| Performance JSON | `tests/dfx/perf/results/result_test_lingbot_world_realtime_{cuda,npu}_*.json` |
| Three standard accuracy results | `tests/e2e/accuracy/results/result_acc_lingbot-world-v2-14b-causal-fast-diffusers_lingbot_world_{precision,wbench,vbench}_*.json` |
| Shared benchmark scoring results | `<output>/results/bench-{precision,wbench,vbench}.json` |
| Per-suite reports | `<output>/{precision,wbench,vbench}.json` |
| Raw scoring details | `<output>/precision/precision.json`, `<output>/wbench/` and `<output>/vbench/` |

`<output>` is `tests/e2e/accuracy/results/lingbot_world/cuda` on CUDA and
`tests/e2e/accuracy/results/lingbot_world/npu` on the NPU evaluation worker. Raw
semantic directories include evaluator logs, input manifests, worker results
and native per-video scores. Capture artifacts include `generation.json`,
`candidate/precision.npz` and each `candidate/<case>/` directory's
`rollout.mp4`, `benchmark.json` and `serve-result.json`, with model, source,
workload and allocation provenance. On CUDA, capture and scoring share the
output directory. NPU capture is uploaded separately under
`tests/e2e/accuracy/results/lingbot_world/npu-capture` and downloaded by the
evaluation step. NPU capture and evaluation outputs are uploaded recursively, preserving native
JSON, logs and any other evaluator diagnostics. The CUDA output also retains
the numerical NPZ and original `candidate/<case>/rollout.mp4` videos. VBench
stages its input symlinks in a temporary directory that is removed after each
evaluation, so the upload tree contains no duplicate staging videos. Input
manifests retain the original video paths. Both CI scoring jobs explicitly set
`ACC_BENCH_RESULT_DIR=tests/e2e/accuracy/results` to match their standard-result
upload path; local runs may override the shared result destination.

Normal measurement or score-floor failures retain their reports before the
pytest assertion, and partial capture retains available inputs and its
manifest. Early setup failure or host termination can prevent complete output.
Score floors come only from reviewed CI measurements; new results are never
auto-promoted to a baseline.
Evaluator setup or measurement failures remain failures with
`comparison_status: unavailable`; they are not reported as measured scores
below the floors. Shared evaluator assets are checked before atomic publication
from a private staging directory; existing corrupt assets still fail validation.

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

For PR builds the NPU LingBot lanes require both `nightly-test` and `npu-test`;
the latter is a separate opt-in for the four-A2 capture and CUDA scoring cost.
The deprecated nightly email distribution is not extended by this PR; the
performance JSON remains available through the existing artifacts/dashboard.
