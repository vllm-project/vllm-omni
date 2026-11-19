# AMD nightly performance qualification

Five existing jobs now require a matching runtime hardware baseline and enforce
the existing 10% regression tolerance. Their grades remain **NonBlocking**.
These are provisional references for qualification, not evidence that the jobs
are ready to become blocking.

## Calibration provenance

Prepared against main `7a92afde03efea84583d716221107899d7ffbd12`. The public AMD nightly
artifacts from November 2–17 report `Hardware: MI300`; ROCm inventory identifies
MI300X. The `MI325` pytest selector does not change that runtime identity.
Only `MI300` references are added. MI325 requires independent measurement.

There were 574 downloaded performance JSON files. Use the **566 results from
successful jobs**, matching 36 distinct configured sweeps. Exclude the eight
partial MiniCPM results from failed November 2–3 jobs. Match each result's test
name and resolved H100 metric map to the corresponding existing configuration,
benchmark-parameter entry and sweep index; preserve all workload parameters.

For each sweep and each existing H100 metric name, the MI300 reference is the
median of successful **November 2–13** measurements (12 observations, or 10 for
MiniCPM). No metric is removed and no tolerance is widened. The four runs from
**November 14–17** are held out from reference selection. CUDA/NPU values and
all existing workload/mark selections remain unchanged.

Historical execution success does not establish performance stability:

| Job | Sweeps/run | Successful executions | Builds with a metric >10% worse than the provisional reference | Held-out builds failing |
| --- | --- | --- | --- | --- |
| Qwen3-Omni No-Async-Chunk Perf | 8 | 16/16 | 8/16 | 3/4 |
| Qwen3-Omni Async-Chunk Perf | 6 | 16/16 | 13/16 | 4/4 |
| Qwen3-TTS Perf | 14 | 16/16 | 13/16 | 3/4 |
| Voxtral TTS Perf | 3 | 16/16 | 8/16 | 3/4 |
| MiniCPM-o 4.5 Perf | 5 | 14/16 | 10/14 | 4/4 |

The counts replay every available sweep independently and count a build once
when any checked metric regresses. They are **offline comparisons**, not
Buildkite runs with this patch. Runtime assertions can stop a failed pytest
case before its remaining sweeps; incomplete artifacts also fail existing
result-count guards. The replay does not control source revision or software
image, so it cannot distinguish software regressions from measurement variance.
In particular, the prepared head includes the newly merged #10312 audio-stream
validation changes, which postdate these historical runs. Qualification must
use the proposed head rather than treating the replay as new-head evidence.

### Source builds

Training: [15194](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15194),
[15246](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15246),
[15309](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15309),
[15393](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15393),
[15480](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15480),
[15528](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15528),
[15562](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15562),
[15626](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15626),
[15697](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15697),
[15818](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15818),
[15983](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15983),
[16075](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16075).

Held out: [16170](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16170),
[16249](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16249),
[16348](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16348),
[16475](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16475).
Find `result_test_*.json` under the named jobs' `benchmark-results` artifacts.

## Buildkite validation and promotion criteria

Maintainers should run these five exact jobs at the proposed head with the
normal AMD image/bootstrap, retaining all existing artifacts. Require:

1. Runtime inventory, source commit, image/dependency versions, allocation and
   physical device identity recorded for each run. Do not substitute a pytest
   SKU marker for that identity.
2. Expected pytest counts (4, 6, 8, 3, 1 respectively) with zero skips and all
   expected result JSONs (8, 6, 14, 3, 5). Preserve the existing cleanup guards.
3. `[baseline-check] START hardware=MI300` and metric pass/fail lines; a missing
   hardware baseline must fail. Missing/invalid metrics also remain failures.
4. Repeated measurements at a fixed source revision and image on an isolated
   allocated node to separate reproducible regressions from variance. Preserve
   prompt counts, seeds, warmups, concurrency and the checked metric set.
5. After addressing observed regressions/variance, calibrate any revised
   references from a declared training set and check separate held-out runs.
   Do not widen thresholds just to turn the historical results green.

Only propose blocking promotion after a fresh stable window (recommended:
seven consecutive scheduled runs with enforcement enabled), complete artifacts
and review of every failure. None of the five jobs currently meets that bar
based on the historical replay alone.
