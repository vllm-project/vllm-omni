# AMD nightly functional and accuracy qualification

These changes prepare three existing **NonBlocking** jobs for native Buildkite
validation. They do not change a job's grade or establish blocking readiness.
Prepared against main `7a92afde03efea84583d716221107899d7ffbd12`.

## Confirmed failures and corrections

- **BAGEL Sleep Mode:** [build 15697, November 10](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15697)
  spent **611 seconds** compiling an AITER attention kernel. The 600-second
  worker RPC deadline expired while the other worker waited on the build lock.
  This job now sets `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=900`, below the fixture's
  existing 1200-second outer initialization timeout. The three sleep tests,
  two-GPU allocation and 165-minute execution budget are unchanged.
- **MiniCPM-o Omni-DuplexEval CI Guard:** the judge's setup and teardown waited
  for global GPU memory to clear while the primary MiniCPM server was alive.
  [Build 16475, November 17](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16475)
  shows two waits of approximately 62 and 61 seconds. The judge now skips this
  global wait; the outer primary server retains it. Absolute checkout-relative
  artifact destinations fix missing performance/accuracy JSON uploads. The
  shared bounded runner adds exact collection/JUnit counts, no-skip validation,
  ROCm inventory and process cleanup checks.
- **MiniCPM-o 4.5 Duplex Function:** [build 15562, November 8](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15562)
  reported a GPU hang and exit 134 during active continuous-input processing.
  Its root cause is **unresolved**. Bounded pre/post ROCm diagnostics now retain
  the agent, host, GPU inventory, driver status and dependency versions. This
  does not hide the original exit code or establish an infrastructure cause.

## Provisional MI300 accuracy floors

All 16 observed Omni-DuplexEval runs, November 2–17, completed the same 66 samples
(30 RTD, 36 PR), concurrency 8, soundtrack mix and protocol
`ca3c122b4d4bf67afd6b18ea5e724b4561bdde48`. Logs identify **MI300X** and result
filenames use `MI300`. No MI325 measurements or baseline are claimed.

Calibration uses November 2–13 only. RTD minima are rounded down to the next
0.1 point; PR allows one fewer success than the observed minimum of 9/36.
November 14–17 are held out from floor selection.

| Metric | Training minimum | Provisional floor | Held-out minimum |
| --- | --- | --- | --- |
| RTD content | 1.8767 | 1.8 | 2.0167 |
| RTD temporal | 1.2944 | 1.2 | 1.4444 |
| PR success | 9/36 | 8/36 | 10/36 |

These floors pass the historical summaries, including all four held-out runs.
They are hardware-specific regression floors, not a claim of CUDA accuracy
parity or fresh validation. CUDA/NPU thresholds are unchanged. The AMD job
requires matching score floors; an actual MI325 run must obtain its own
calibration instead of silently passing without a quality gate.

Source builds: [15194](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/15194),
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
[16075](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16075),
[16170](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16170),
[16249](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16249),
[16348](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16348),
[16475](https://buildkite.com/vllm/vllm-omni-amd-ci/builds/16475).

## Maintainer-triggered Buildkite validation

Run the three exact job labels above at the proposed commit, keeping their
existing nonblocking grades. Use the normal AMD image/bootstrap, record the
commit and image identity, and retain each job's `artifacts/rocm-*` directory.

1. BAGEL: require all three tests with zero skips. Include a naturally cold
   AITER-cache run on an isolated CI pod; do not clear a shared runner cache.
   Confirm warmup finishes before 900 seconds without collective/startup errors.
2. Omni-DuplexEval: require one executed test, 66 published/evaluated samples,
   zero failed/skipped samples, all three MI300 floors enforced, and both JSON
   result types uploaded. Confirm judge memory-wait warnings disappear and the
   primary server's final cleanup and process-leak check succeed.
3. MiniCPM Duplex Function: require all six nodes, zero skips and no aborts.
   Correlate the November 8 failure on `gpu9106` with host/driver RAS records;
   repeat on another allocated host. Successful repeats alone do not classify
   the original hang as infrastructure.

Recommended promotion evidence: seven consecutive scheduled runs with the
new guards enabled, plus the cold-start/failure-specific checks above. Failures
reset that observation window. Maintainers decide promotion after reviewing
matching-head native evidence; CPU tests alone cannot qualify these jobs.
