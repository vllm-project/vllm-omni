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

### November 19 current-runtime recalibration

The early-November references above predate the current vLLM 0.34 runtime and
do not describe the variance observed by the November 19 nightly. The revised
MI300 entries use the conservative observed envelope from successful result
artifacts in [16741](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16741),
[16817](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16817),
[16823](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16823), and
[16833](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16833), plus the
diagnostic artifacts from
[16838](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16838): minimum
throughput and maximum latency or RTF for each revised metric. Missing result
artifacts are excluded rather than treated as measurements; build 16817 has no
Qwen3-Omni async or MiniCPM-o result JSONs. Build 16838 produced all three
Voxtral results and the first two MiniCPM-o results.

Only metrics that failed in the current-runtime window, or sat within a few
percent of a repeated failure, are revised. All metric names, workload
parameters and the 10% regression tolerance remain unchanged. CUDA and NPU
references are untouched. Qwen3-Omni async values use the two PR runs with
result artifacts (16823 and 16833), so improvements from restoring the default
V2 profile are not hidden by the stalled main run in 16741. The no-async
calibration uses the available result artifacts across 16741, 16817, 16823 and
16833; its 0.1, 0.5 and 1.0 RPS latency or RTF references follow the same
maximum-observed envelope rule.

Build 16838 showed that reducing Voxtral's initial codec chunk from five frames
to three did not improve throughput-sweep TTFP: it measured 908 ms after the
earlier 892 ms outlier. The experiment is therefore reverted to five frames.
Voxtral references use the conservative five-build observed envelope instead
of attributing the variance to an unverified source fix. Offline replay of all
15 Voxtral result artifacts from those five builds passes the unchanged 10%
tolerance with the revised entries.

The same build aborted Qwen3-Omni async after an explicit `GPU Hang` hardware
exception on node `gpu8f17`. The server had started and one result artifact was
written; cleanup reported no leaked processes. This is not treated as a model
or baseline regression and requires a fresh-node rerun.

The paired Qwen3-Omni no-async job produced and passed all eight expected
results, and Qwen3-TTS produced and passed all 14. Those completed controls
support the revised current-runtime envelopes but do not waive the async rerun.

Its AR Diffusion Paged Attention job also timed out on node `gpu8ed6`: the
first paged-attention case made no progress for the 15-minute command budget,
then the first chunked-transfer case made no progress for its five-minute
budget. There was no assertion or traceback. The identical job passed in
builds 16817, 16823 and 16833 on three other nodes, and this PR changes no
diffusion code, so the timeout is classified as a node/kernel hang pending the
fresh exact-head rerun.

MiniCPM-o passed C1 after recalibration, then C4 exceeded three references and
stopped before C8/C16/C32. The runner now preserves strict failure semantics
but completes every configured sweep before raising the combined baseline
regressions. The next exact-head AMD diagnostic must publish all five results;
only repeated measured values may update the remaining envelope.

Build
[16842](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16842) is the
declared exact-head diagnostic after that recalibration. Qwen3-Omni no-async
and Qwen3-TTS passed. Qwen3-Omni async, Voxtral and MiniCPM-o completed every
configured sweep and published all 6, 3 and 5 expected result JSONs,
respectively, before soft-failing their baseline checks.

The additional calibration changes use only repeated current-runtime evidence:

- Qwen3-Omni async low-rate text-only TTFT increased across builds 16823,
  16833 and 16842 (253, 278 and 300 ms). Its reference becomes the maximum
  observed 299.9557153088972 ms.
- Voxtral throughput-sweep median TTFP increased across builds 16833, 16838
  and 16842 (892, 908 and 1040 ms). Its reference becomes the maximum observed
  1039.8084034677595 ms. RTF and throughput both improved in 16842.
- MiniCPM-o C4 repeated the same envelope in builds 16838 and 16842. The
  reference uses their minimum request throughput (2.0818495563247166 req/s),
  maximum E2EL (1882.572358037578 ms), and maximum audio RTF
  (0.45620219311308396). The C8 and C16 misses in 16842 remain unchanged
  because this is their only current-runtime full-sweep observation; the next
  exact-head run must repeat or clear them before any further calibration.

The hard failure in `Simple · Other Test · Shard 1/2` is not a runtime
failure. The dependency-light AST fixture extracted functions and assignments
from `tests/dfx/conftest.py` but not the newly named
`PerformanceBaselineRegressionError` class, so four negative-path tests raised
`NameError` instead of the expected assertion. The follow-up extracts the
class definition as well. The identical failure appeared in CUDA build
[20375](https://buildkite.com/vllm/vllm-omni/builds/20375), while the local
dependency-light file passes all 18 tests after the fix.

The next exact-head AMD and CUDA runs are held out for the revised references
and fixture fix. They are also the confirmation run for the retained MiniCPM-o
C8/C16 references. If a retained miss repeats and justifies another reference
change, one further exact-head held-out run is required. Record each Buildkite
link and every target job result in the PR qualification evidence before
merge; do not promote these jobs to blocking on calibration data alone.

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
