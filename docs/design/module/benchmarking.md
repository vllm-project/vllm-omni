---
title: Benchmarking
kind: module
status: draft
owners:
  - "@alex-jw-brooks"
  - "@Bounty-hunter"
primary_code_paths:
  - vllm_omni/benchmarks/**
  - vllm_omni/entrypoints/cli/benchmark/**
related_code_paths:
  - benchmarks/**
depends_on:
  - entrypoints.md
  - profiling.md
validation_paths:
  - tests/benchmarks/**
  - tests/dfx/perf/**
upstream_refs:
  - vllm.benchmarks
last_reviewed: 2026-07-16
---

# Benchmarking

Benchmarking defines reproducible workloads, metrics, warm-up, measurement,
and result metadata for comparing vLLM-Omni performance.

## Candidate invariants

### BENCH-INV-001: Results are reproducible

**Rule:** Results MUST identify the model, revision, commit, hardware,
configuration, workload, concurrency, warm-up, and measurement window.

### BENCH-INV-002: Correctness precedes performance

**Rule:** A benchmark MUST validate successful and semantically acceptable
outputs before using timings as performance evidence.

### BENCH-INV-003: Metric definitions are explicit

**Rule:** Metrics MUST state their unit, population, and aggregation.

## Safe-change guide

Test workload parsing and metric calculations with deterministic fixtures. Do
not compare materially different workloads or runtime settings.

The [LingBot World regression suite](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/lingbot_world/README.md)
wraps the served realtime workload in blocking pytest jobs. It measures a frozen
reference on the same hardware/dependencies, calibrates from three independent
server starts, and checks candidate performance separately. Fixed-input VAE
replay and instrumented lifecycle/memory checks run outside timing windows.
The PR smoke executes its recorded chunk/warmup/frame-rate/timeout settings
explicitly, validates raw output with the shared metric checker, and retains
those metrics. CLI defaults must not silently change the recorded workload;
short smoke timings are functional diagnostics rather than calibrated limits.
The CPU gate also executes uploader and AMD pipeline contract tests. Shell
logging and device-reservation changes must preserve selected tests, shard
coverage, blocking status and infrastructure-only retry rules.
CUDA generation may retry a lost Buildkite agent once while retaining the
original attempt. Numerical/quality failures, OOM, cancellation and timeout
remain blocking; consumers validate the current attempt's complete artifacts.
Calibration reuse requires recorded optimization settings: its nonempty
`engine_environment` string mapping must match the measured contract. Missing
settings are invalid even when both records omit the field.
Reference-noise calibration also binds recorded playback/mux frame rates and
both the measurement warmup and inference warmup lengths. Missing or changed
values reject reuse. Fresh profile-owned records also bind a canonical digest
of the complete measured profile; other scenes and rollout lengths need their
own qualified calibration.
Nightly CUDA uses two recorded-zero H100 reference cohorts and pinned-decoder
pixel summaries. Its fixed population owns reference-spread ceilings; loading
that policy must preserve global weekly/NPU limits and candidate tolerances.
The nightly pixel population records all 15 unordered comparisons among its six
reference videos, including cross-cohort pairs. Version 2 validates both
three-start reference cohorts and the complete ordered pair list while retaining
six native scores per dimension. Version 3 additionally requires complete
per-case reference-spread limits. Fresh-reference and candidate checks use the
selected case's limits, preserving candidate tolerances and other profiles.
Reference-only held-out comparisons and actual
reencode, localized-degradation and truncation controls qualify the pixel method;
they do not turn earlier failed candidate jobs green or replace final-source CI.
Historical unrecorded records remain separate diagnostic fixtures. Fitted
calibration and negative controls do not replace fresh target-hardware acceptance.
Calibration compatibility errors retain raw measurements, numerical results and
quality scores for review; candidate comparisons stay unjudged and block acceptance.
CI pixel comparisons execute in the same verified dependency environment as
native evaluators, using isolated Python workers with retained versions, logs and
incomplete results on failure. Fixed reference pixel summaries require recorded
decoder dependency versions matching that environment; missing metadata rejects reuse.
The two pinned OpenCV distributions share `cv2`. Fresh provisioning selects the
headless provider after dependency sync and verifies its imported files against
wheel `RECORD` hashes. Include provider-verification code in the evaluator cache
key; validate completed caches without reinstalling or rewriting their provenance.
Retain the verified provider in quality artifacts and require each pixel worker's
build fingerprint to match it. Package-version equality alone does not prove
which distribution owns the loaded native module.
Weekly generation uses six case shards, each with three of the 18 fixed cases.
Quality evaluation splits each generation shard into three required case jobs,
so each GPU evaluator scores six videos under the unchanged job deadline. The
CUDA three-case evaluator exceeded its 240-minute budget. Preserve all 18 cases,
three independent starts per role, rollout lengths, warmups and regression
budgets. Record the evaluated case while retaining the complete producer
profile and manifest hash; verify all producer artifact hashes and preserve the
producer's failure in the final gate. Use distinct per-case output directories.
The initial CUDA/NPU production reference is frozen at
`48cebe5a248d4b0160f20c7fc18b5d1c46883872`, based on Omni `v0.32.0rc1`
and the NPU offload/compiler-context fixes. The CPU gate exercises compiled
attention graph breaks, exact eager/compiled outputs, K/V reuse and leased
output lifetimes. Require vLLM 0.32.0 and verify both source checkouts
can import their serving entrypoints before provisioning model weights.
A compatible Ascend image is required before NPU collection can resume.
Fresh calibration is pending:
older-reference records remain immutable and must fail contract validation
against the new pin. Collect fresh reference-role data, freeze training records,
then validate independent held-out references and final-source candidates.
Historical candidate artifacts cannot substitute for new references. Native
reference-spread calibration is scoped per platform, profile, case and dimension;
reference calibration cannot change candidate regression allowances.
NPU weekly collection separates each case shard's three independent
reference/candidate pairs into bounded jobs. A mandatory CPU gate verifies the
complete matching population per shard before quality evaluation; missing or
cross-shard pairs fail that gate. Partitioning jobs must preserve all cases,
sessions, chunks, starts, hardware and regression budgets.
LingBot regression jobs share organization-wide concurrency groups across builds
and triggers: one CUDA generation/smoke job, one NPU collector, two CUDA/NPU
quality jobs combined, and one L4 worker for CPU unit/aggregation jobs. Keep
group names independent of build, branch, schedule and matrix coordinates.
All regression jobs use priority 0, the same as normal PR tests.
On PR builds, NPU regression groups also require the explicit `npu-test` label;
the shared nightly/weekly label alone cannot admit NPU collectors or evaluators.
Scheduled non-PR nightly/weekly admission remains unchanged. Check the actual
rendered group condition rather than relying on external pipeline admission.
Use eager admission so dependency-blocked jobs do not impose ordered-group
barriers on ready jobs. Limits bound peak reservation, preserve the full workload,
and do not preempt running work or replace hardware acceptance.
GPU evaluators consume checksum-verified CUDA or NPU videos and enforce native
per-dimension quality limits. Persist evaluator hardware, dependencies and input
hashes before native workers start, so forced termination retains a failed report
with its provenance. Incomplete populations, unstable references,
missing prerequisites and skipped selected CI tests cannot establish acceptance.
Generation, aggregation and quality record their actual candidate source revision
in the initial failed report before input validation or model work. Incomplete
artifact handoffs must retain the consumer's own source identity, distinct from
the revisions of any copied producer files.
Failed server initialization retains a bounded process-state snapshot in logs
before cleanup, outside successful measurement paths. Unavailable process and
Linux wait-channel data remain explicit without hiding the original failure.
After successful server teardown, LingBot smoke and full collectors assert the
shared 5% engine-memory cleanup condition for every selected logical device.
Use a zero wait budget outside performance, retain a failed checkpoint before
querying, and propagate device errors or residual allocations to pytest. Shared
memory-wait deadlines use a monotonic clock; a clock correction cannot extend
the cleanup assertion.
