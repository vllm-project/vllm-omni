# Wan TeaCache evidence summary

**Experimental draft; production acceptance has not passed.** No default Wan
TeaCache profile is enabled. This summary includes the completed 2026-10-10 evaluation jobs, checked on
2026-10-12. It does not validate subsequent source changes, including the
branch-local coefficient-collection change in `8dc594b5`.

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
- PP2/CFG1 (retry job 1096862) and PP2/CFG2 (job 1096101) each completed
  36 paired videos: mean/worst SSIM 0.974147/0.938729 and temporal error
  0.250010. Cached/native latency ratios were 1.066345 and 1.101729.
  Both failed temporal and latency gates. These results postdate the linked
  artifact snapshot; raw files remain in the shared validation directory.
- Full-compute smoke equivalence passed for all four PP/CFG topologies at
  256x256, 5 frames, 6 steps. This is not quality acceptance.

The harness uses stage-specific coefficients, negative-prediction clamping and
an observed-input-range guard absent from the production policy. Before marking
this PR ready, qualify a profile through the unmodified production hook with
paired quality measurements and complete end-to-end timing. Harness latency
excludes loading, warmup and export and includes tracing overhead. No serving
throughput claim is made: that requires a concurrent-arrival run reporting
actual admission and batch behavior.

## Final matrix and audit

All four topologies completed on `ef6379755f1f12a64a41336efa70b62e43b933f6`:
144 native/cache pairs (288 videos), BF16, 512x512, 17 frames, 50 steps,
guidance 5.0. All four pass both SSIM gates and fail temporal-error and latency
gates: **8 gate passes, 8 gate failures, 0 skips**. These are measured acceptance
gates, not unit-test counts. No production default profile is promoted.

Each PP2 run recorded stage-0 positive/negative hits of 168/168 and stage-1
hits of 281/279. These are actual local block-stack skips. All 36 requests per
branch began at step zero with full computation; branch states were distinct
and paired final latents were finite. The stricter post-run audit passed.

Original PP2/CFG1 job 1096100 timed out without complete video pairs, repeatedly
reporting a ShmRingBuffer shared_memory AttributeError from the result pump.
Retry 1096862 preserved the source, model, profile and environment and completed
in 8m41s with a 30-minute limit. Root cause of the first failure remains
unresolved; a successful retry does not prove it fixed. Shutdown resource-tracker
cleanup warnings remain in successful logs.

PP2 raw evidence and four-gate JUnit reports are preserved at:

```text
/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/wan-teacache-validation/
  validation-1096100/  # unsuccessful attempt
  validation-1096101/  # PP2/CFG2
  validation-1096862/  # PP2/CFG1 retry
```

Each completed directory contains `quality.xml`, `held-out/audit.json`,
`held-out/cache/comparison.json`, rank traces, native/cache videos and frames
0/8/16. Generated artifacts remain outside the source tree.
