---
name: inference-perf-optimization
description: Choose, test, and reassess inference serving optimizations for latency, throughput, streaming continuity, and resource use. Use when attributing a regression, estimating whether a candidate merits implementation, or revisiting an experiment after the bottleneck changes.
---

# Inference performance optimization

Start from a reproducible run of the target workload. If none exists, measure the primary metric with representative requests through the real serving endpoint. For a regression, reproduce the difference on known good and bad revisions before narrowing the change. With a credible candidate, test the assumption most likely to change the decision. If the bottleneck is unknown, compare slow and typical requests or identify the saturated resource.

## Freeze the comparison

- Choose the workload, primary metric, aggregation, and acceptance criteria. Examples include mean time to first audio at concurrency one, P95 real-time factor (RTF) at target load, and audio seconds delivered per second. Track affected constraints such as first and last packet latency, peak memory, completion rate, and quality.
- Freeze the server, request, and benchmark commands or equivalent manifests. Record the loaded revision, image, model, precision, effective configuration, device and replica placement, request set, concurrency, warmup, and sampling protocol. For queueing tests distinguish fixed arrival rate from closed-loop load. Account for client submissions, ingress waits, and actual service concurrency. If versions, hardware, and load all change, report the whole-stack difference.
- Separate cold startup, warm steady state, and first capture of a new shape during normal traffic. Keep the representative request mix. Use heavyweight profiling for diagnosis. Check completion counts and output validity. If paired audio durations differ, report original latencies, durations, RTF, and quality. Recomputing with fixed reference durations can help explain the difference.

## Locate the critical path

- Align client events, stage timings, and runtime counters for the same request. Distinguish submission, device completion, and client receipt. Separate compute, queueing, synchronization, transfer, and delivery. Use [worked cases](references/worked-cases.md) when counters and wall time suggest different explanations.
- Specify the change, eligible requests, work removed, additional cost, and plausible impact on the primary metric. Compare the optimistic and realistic estimates with the smallest useful gain and baseline variability. If the choice hinges on an unknown, measure it first. See [candidate selection](references/optimization-directions.md).
- Check existing fast paths. Look up earlier experiments by stage, mechanism, and retest condition. After a baseline change, remeasure the limiting path and rank affected deferred candidates alongside new ones. For batching, count ready rows before grouping, compatible rows after grouping, effective batch width, and fallback reasons.
- For a ceiling estimate, report measurements alongside device-calibrated estimates per workload. State precision, batch width, cache state, placement, and serial dependencies. Leave unexplained headroom unattributed until resource saturation and end-to-end timing support a bound.

## Test the candidate

- Start with an eligible request and verify execution through replay and fallback counts, effective batch width, device branch, communication volume, or synchronization counts. Toggle one variable on the same revision or compare a patch containing only the intended change.
- On a miss, regression, or error, preserve the run. Identify the first missing link between activation, work removed, and service benefit. Use [candidate reassessment](references/candidate-reassessment.md) to choose the next test. For adoption, use independent paired runs, alternate order where practical, and report all trials and variability. Request-level pairing handles input differences. Independent runs reveal process or batch drift. To rule out a useful gain, the uncertainty interval must exclude the minimum useful effect. Attribute a combined result to individual changes only after ablation.
- Verify completion, output volume, termination, and request isolation. For changes to precision, sampling, length, step count, or cache semantics, include affected quality and boundary cases. Bind counters and quality evidence to the arm actually tested. See [multi-stage audio and Ascend](references/omni-ascend.md) for specialized checks.

## Decide and retain the evidence

- Distinguish an invalid comparison, inactive path, regression, correctness failure, unresolved gain, useful gain excluded for this workload, gain pending qualification, and recommendation to adopt. Retain crashed or truncated runs as failure evidence. Evaluate continuity and OOM fixes against their own targets. For stacked changes, measure the increment against the latest accepted baseline and cumulative effects against the original qualified baseline.
- Compare predictions with measured eligibility, unit cost, overlap, queueing, and new overhead. Update candidates sharing the failed assumption. Record experiment outcomes separately from deployment status. If a batch exits abnormally, check the reason and artifact completeness before using completed subtests.
- Record the premise, arm identifiers, mechanism evidence, per-run metrics, quality, and artifacts. For a deferred candidate, save the blocker, observable retest condition, and reproducible entry point. Close a mechanism only for the tested conditions.
- Deliver once the target and its constraints pass. If the remaining gap exceeds what the current family of changes can deliver, return to the unexplained slow interval or saturated resource. If no affordable experiment would change the decision, leave the latest qualified version, remaining gap, and conditions for reopening.
