# Host starvation: from exposed gap to source owner

## Establish the symptom

Select repeated or high-impact gaps after the coverage audit. Correlate CUDA API
and GPU events with the CPU thread/process that prepares the next operation.
Separate no ready request, no ready input and no timely submission. Positive GPU
queue time by itself often means work was supplied in advance.

Write an observation chain, for example: prior device work ends -> preparation
continues on worker thread -> launch occurs -> next kernel starts. State missing
edges. Timestamp coincidence without the same request/dependency is insufficient.

## Root-cause branches

| CPU behavior during exposure | Evidence to collect | Recommendation and falsifier |
| --- | --- | --- |
| Running Python repeatedly | Sampled stacks, per-step call counts, source values | Consider invariant preparation/less object work; refuted if values change or work is hidden |
| Running native preparation | Native stack, tensor metadata/packing, allocator events | Inspect actual costly owner and copies; do not blame Python from its enclosing frame |
| Waiting on GPU | CUDA synchronization correlation and consumer need | Follow synchronization guide; required GPU result is not CPU inefficiency |
| Waiting on IPC or lock | Producer identity, lock owner, queue timing and request IDs | Inspect producer/backpressure; removing synchronization may violate ordering/lifetime |
| Runnable but descheduled | CPU scheduling/context-switch samples, cgroup quotas, other load | Recommend resource/config investigation, not a kernel rewrite |
| High CPU use with low progress | Thread-pool contention, oversubscription, GIL-relevant Python stacks | Propose bounded thread/config experiment; more threads are not automatically better |
| Allocations on repeated path | Allocation count/size, CPU/GPU allocator, lifecycle | Inspect reuse at the owner; storage sharing must preserve request isolation |
| Event loop occupied by output work | Encode/serialize duration, queue growth, other request delays | Diagnose output ownership and supported async path; do not change API completion semantics |

Use progressively heavier evidence: timeline -> targeted CPU sampling -> selected
stacks/shapes -> small source-level reproduction when authorized. Full-stack
profiling on every request can manufacture the bottleneck. If symbols or sampling
permissions are absent, keep the owner unresolved rather than guessing a function.

## Host exposure accounting

Report host preparation total, overlap with device activity, observed exposed
intervals and the dependency evidence. Do not subtract all host duration from E2E.
Do not treat a sleep/poll wait as removable until the awaited state is understood.
For multi-stage workloads, readiness may come from another process/device; return
to the L1 path map rather than analyzing the worker in isolation.

## Recommendation card example

Suppose repeated step preparation ends after the preceding kernel, and samples
identify request-invariant metadata reconstruction. Propose testing relocation or
reuse at that metadata owner, with exact validity conditions. The predicted gain
is limited by exposed preparation under the fixed schedule, not total Python time.
Acceptance would require unchanged request/step semantics, unprofiled E2E evidence
and no added cross-request mutable state. This is a hypothesis until tested.

## When to stop this branch

Stop deep CPU sampling once the expensive exposed paths are explained or collection
cannot resolve them. Report low current leverage or the concrete evidence gap.
Do not infer high kernel efficiency, or absence of future host bottlenecks after
device-side acceleration, from a low-exposure result.
