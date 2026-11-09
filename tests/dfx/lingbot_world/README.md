# LingBot World CI cases

These are real-model CI definitions, not a delivered benchmark campaign or a
claim that a particular revision has passed on hardware. CUDA and NPU use the
existing test images and queues. No model execution is required to edit or
review these definitions.

## Independent tests

| Case | Pytest entrypoint | Required inputs |
| --- | --- | --- |
| Numerical accuracy | `test_precision.py::test_numeric_precision` | Paired VAE samples and complete generated videos |
| WBench | `test_wbench.py::test_wbench_quality` | Paired videos and pinned WBench evaluator |
| VBench | `test_vbench.py::test_vbench_quality` | Paired videos and pinned VBench evaluator |
| Performance | `test_performance.py::test_performance_regression` | Original chunk-arrival records |
| Long stability | `test_stability.py::test_cuda_long_stability` or `test_npu_long_stability` | Candidate server, two 600-chunk sessions |

Collection does not judge numerical accuracy, speed, WBench or VBench. Each
consumer checks its own complete input population and writes its own report:
`precision.json`, `performance.json`, `wbench.json` or `vbench.json`. There is no
combined semantic verdict. A WBench regression does not prevent the other
cases from running. An incomplete producer still fails its CI job; consumers
may inspect authenticated available artifacts and fail on their own missing or
invalid inputs. No consumer reads another consumer's result.

## Real paired measurements

`test_collect.py` starts a clean, pinned official reference checkout and the
candidate on the same runtime, deployment and locked inputs. The reference is
`e96e3beeb6fa0842ad5f4a74b4d409edca39a0d5`; neither role changes dependency
versions. Three independent starts alternate reference/candidate ordering.
The profiles retain the existing 40-chunk cases, warmup exclusions, frame size,
seeds, camera trajectories and measured-session counts. NPU pairs may be split
across three jobs to respect existing job deadlines; each uses the same
realtime-video backend with seven measured sessions per case and one
eight-chunk warmup. The aggregate only checks and assembles the recorded inputs.
It never judges semantic acceptance. Selected-source CWD and PYTHONPATH retain
the NPU compiler paths, and the profile's one CPU thread per worker remains in
the server environment.

Collection calls the same `vllm bench serve --omni` realtime-video lifecycle
for each case. Each arm writes `precision.npz` and, per case, canonical
`serve-result.json`, its source-bound `benchmark.json` domain projection, and
`rollout.mp4`. The projection preserves the existing independent-judgment
contract; it does not schedule requests or recompute a second benchmark. The
manifest records source, profile, deployment, runtime and artifact hashes.
Lossless packaged realtime-video remuxing preserves encoded packets and
timestamps; it does not modify the product media implementation.

Performance recomputes metrics from raw chunk arrivals. The first six chunks
and the terminal chunk do not enter steady cadence metrics; warmup sessions do
not enter measured populations. The original CV and 5% relative-regression
limits apply to the current run's three reference starts. The 12 FPS target is
the playback/SLO measurement basis, **not** an absolute 12 FPS acceptance gate.
P99 is reported but only gated with at least 1000 steady observations.

Numerical accuracy retains exact VAE replay input/output comparisons and
decoded PSNR/SSIM checks over the complete videos and aligned time windows,
including uniform-frame rejection. WBench and VBench retain their separate
continuous-dimension, binary-activity and reference-spread limits from
`quality.json`. Both semantic suites validate complete video frame coverage;
VBench also verifies complete bounded background-encoder inference. Native
scorers run in the isolated, pinned evaluator environment.

## Long stability

The CUDA/NPU long profiles use two candidate-only 600-chunk sessions. They
record actual camera acknowledgements and post-warmup memory samples, retain
the 256 MiB per-device growth bound within and across sessions, reuse the
existing streamed-block/paced-playback/stop lifecycle checks, and require
server/device cleanup. These instrumented measurements are not performance
samples. They do not repeat reference, tensor or semantic-quality campaigns.

## Commands and local verification

The CI launcher supports:

```bash
bash tests/dfx/lingbot_world/run_ci.sh generate tests/dfx/lingbot_world/configs/cuda_nightly.json OUTPUT
LINGBOT_MEASUREMENT_REPEAT=0 bash tests/dfx/lingbot_world/run_ci.sh collect tests/dfx/lingbot_world/configs/npu_nightly.json PAIRS/pair-0
bash tests/dfx/lingbot_world/run_ci.sh aggregate tests/dfx/lingbot_world/configs/npu_nightly.json OUTPUT PAIRS
bash tests/dfx/lingbot_world/run_ci.sh precision PROFILE OUTPUT GENERATION
bash tests/dfx/lingbot_world/run_ci.sh performance PROFILE OUTPUT GENERATION
bash tests/dfx/lingbot_world/run_ci.sh wbench PROFILE OUTPUT GENERATION
bash tests/dfx/lingbot_world/run_ci.sh vbench PROFILE OUTPUT GENERATION
bash tests/dfx/lingbot_world/run_ci.sh stability tests/dfx/lingbot_world/configs/npu_long.json OUTPUT
```

NPU collection needs repeats 0, 1 and 2. Existing resource declarations reserve
eight devices and expose four to generation/stability; artifact-only numerical
and performance checks use CPU, and each semantic evaluator uses one H100.
Runtime imports, locks and fixtures must be available; missing prerequisites
are failures in selected CI cases. Local native tests require explicit
`--lingbot-profile`; absent profiles skip locally rather than loading models.

CPU tests under `tests/benchmarks/test_lingbot_world_*.py` exercise metric math,
malformed populations, native-worker boundaries, independent failures, video
coverage and fixture contracts. CPU checks, pytest collection and CI rendering
do not establish real-model accuracy, speed or stability results.
