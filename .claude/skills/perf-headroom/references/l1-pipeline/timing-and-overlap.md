# Wall time, summed work and critical-path impact

## Definitions

- **Phase wall time:** elapsed time from the declared start to actual completion,
  including waits. A Python submission range is not GPU completion.
- **Summed kernel time:** sum of durations of selected kernel events. Concurrent
  kernels each contribute their duration, so the sum may exceed wall time.
- **Device busy union:** duration of the union of recorded kernel/copy/set events
  on one device and timebase. It is activity, not full-resource useful utilization.
- **Exclusive attribution:** disjoint intervals assigned under a stated rule.
  Cross-category overlap needs its own bucket or a justified allocation.
- **Critical-path impact:** change in completion after dependencies and resource
  ordering are recomputed. It is not an ordinary event-table column.

Name the device, request population, window and denominator for every number.
Two overlapping stages and two GPUs are not additive E2E buckets. An exclusive
accounting bucket is not automatically removable or causally critical.

## Synthetic example: why a large component can be hidden

Two independent kernels run concurrently without resource interference. A lasts
10 ms and B lasts 8 ms; both must finish before the phase completes.

```text
time       0         8  10 ms
A          [----------] 10 ms
B          [--------]    8 ms
completion             ^
```

Summed kernel time is 18 ms, busy union is 10 ms and phase wall time is 10 ms.
Making B take 4 ms reduces the sum to 14 ms but does not change completion.
Making A take 7 ms while B stays 8 ms gives an 8 ms phase: the limiting branch
switches to B. If B depends on A instead, the original wall time is 18 ms.

The bundled `replay` helper can represent the concurrent case as:

```json
{
  "resource_order_confirmed": true,
  "nodes": [
    {"id": "A", "ms": 10, "deps": []},
    {"id": "B", "ms": 8, "deps": []},
    {"id": "join", "ms": 0, "deps": ["A", "B"]}
  ]
}
```

This is a synthetic sensitivity model, not a measured trace or hard bound. In a
real graph, include release times, CPU/device/stream ordering and data edges;
do not remove resource edges merely to create a shorter prediction. Shared
bandwidth, SM contention and locality can change node durations when overlap changes.

## Manual accounting procedure

1. Choose an explicit request/phase interval in one trace timebase. For device
   activity, clip events to it and discard only events entirely outside it.
2. Record the event sum separately. Union all activity intervals; compute head,
   internal gaps and tail. Keep unlabeled activity, rather than treating it idle.
3. Audit event completeness and graph nodes before interpreting uncovered time.
   An envelope is neither proof of internal busy time nor proof of idle time.
4. Attribute overlapping categories explicitly. The helper retains a
   `cross_category_overlap` bucket instead of counting it twice.
5. Inspect dependency and launch correlation for high-cost and no-progress regions.
   Only then estimate exposure or an alternative schedule.
6. Report baseline wall time alongside diagnostic-window time, with capture
   perturbation. They need not match and must not be forced to close together.

The helper's `summed_event_ms` includes copies/sets as well as kernels. Its
`overlap_excess_ms = summed_event_ms - busy_union_ms` is duplicate-counted work
duration, not necessarily the length of time with concurrency: three simultaneous
events contribute twice their shared duration. Neither field is an E2E saving.

## Common misinterpretations

- A 300 ms host synchronization may simply wait for required device work; its
  call duration is not 300 ms of synchronization overhead.
- CPU preparation lasting 5 ms entirely inside independent 10 ms GPU work has
  no direct exposed delay in the fixed-duration model. A 15 ms preparation could
  expose 5 ms if the next GPU work requires that preparation and the first kernel.
- Long kernel queue time can mean healthy producer lead, not starvation.
- Hidden work still consumes resources. Reducing it may improve contention,
  energy or loaded-service capacity; that is a separate hypothesis to measure.
- A throughput-oriented overlap change can worsen individual latency. Keep the
  objective from L0 rather than declaring all additional overlap beneficial.
