# Wan TeaCache evidence summary

**Experimental draft; production acceptance has not passed.** No default Wan
TeaCache profile is enabled. This summary describes the 2026-10-10 snapshot,
not validation of subsequent source changes.

Generated videos, frame images, rank traces, machine snapshots, fitted profiles
and JUnit reports are excluded from the current source tree. The original
[full evidence archive](https://github.com/fusheng-ji/vllm-omni/tree/cbe2bf638c36619fa54bf7d492223e4252011f80/tools/wan_teacache_validation/results)
and [environment snapshot](https://github.com/fusheng-ji/vllm-omni/blob/cbe2bf638c36619fa54bf7d492223e4252011f80/tools/wan_teacache_validation/requirements.lock.txt)
remain available at the immutable reporting commit. See the
[reproduction entry point](../README.md), [pinned model](../model.json) and
[calibration/held-out prompts](../prompts.json).

- Independent evaluation runtime: `ef6379755f1f12a64a41336efa70b62e43b933f6`.
- All 18 completed calibration candidates failed at least one acceptance gate.
- PP1/CFG1 and PP1/CFG2 each evaluated 36 paired videos: mean/worst SSIM
  0.988863/0.971094 and temporal error 0.163825. Cached/native latency ratios
  were 1.027440 and 1.044654, respectively. Both failed temporal and latency gates.
- PP2 held-out evaluation was pending at that snapshot; queued jobs are not passes.
- Full-compute smoke equivalence passed for all four PP/CFG topologies at
  256x256, 5 frames, 6 steps. This is not quality acceptance.

The harness uses stage-specific coefficients, negative-prediction clamping and
an observed-input-range guard absent from the production policy. Before marking
this PR ready, qualify a profile through the unmodified production hook with
paired quality measurements and complete end-to-end timing. Harness latency
excludes loading, warmup and export and includes tracing overhead. No serving
throughput claim is made: that requires a concurrent-arrival run reporting
actual admission and batch behavior.
