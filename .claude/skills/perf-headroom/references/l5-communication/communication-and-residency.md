# Distributed communication and data residency diagnosis

## Topology and baseline

Record GPU/rank mapping, node/NUMA placement, interconnect links, peer access and
transport implementation. Inspect the active TP, Ulysses/Ring SP, CFG or VAE group
configuration, not just the total device count. Distinguish replicas serving
independent requests from ranks collaborating on one request.

For a reference, use the same device group, operation, message size, dtype and
completion semantics. A matched communication microbenchmark is an empirical
reference, not a hard physical limit. Its algorithm/bus bandwidth convention may
differ from raw payload bytes divided by time; record the exact definition.

## Readiness versus transmission

For a collective or handoff, map:

```text
producer computation -> local ready -> communication -> remote visible
                                                   -> consumer dependency -> use
```

Compare entry times across ranks. If one rank arrives late, investigate its input,
compute, CPU submission and previous collective. If all arrive together but exchange
is slow, examine topology, message granularity, staging, contention and backend.
A busy-waiting GPU kernel can make the busy union high while useful work stalls.

## Exposure and resource competition

Record transfer spans and overlapping independent computation. Then identify which
completion edge delays the consumer. The intersection of two spans is descriptive
overlap, not automatically successfully hidden cost. With shared HBM/SM/link
resources, concurrent durations may differ from isolated durations.

If allowed and necessary, propose bounded matched isolated/combined experiments to
separate latency hiding from interference. Do not silently change a live schedule.
Apply joint reasoning when faster local compute exposes more communication.

## Parallelism-specific questions

| Mode | What to distinguish |
| --- | --- |
| TP | Compute reduction versus smaller GEMMs, collective frequency, reduction/packing and rank skew |
| Ulysses / Ring SP | All-to-all or ring exchange, sequence/head divisibility, attention work partition, packing and overlap |
| CFG parallel | Branch balance, duplicated setup, synchronization and final combine; actual CFG execution semantics |
| VAE patch parallel | Tile borders, exchange/assembly, decoder imbalance and required overlap |
| Stage transport | Serialization, staging copies, chunk size, sender/receiver visibility and consumer backpressure |
| Request replicas | Queue routing, shared CPU/output resources and throughput/SLO rather than single-request speedup |

Reject a topology change as a free local optimization: it may alter capacity,
cost, kernel shapes and the workload objective. Recommend it as a separately
scoped experiment with exact resource and quality constraints.

## Offload and residency

Measure peak live/reserved/device memory separately. Identify what is resident,
what crosses CPU/device or peers, and how often. Inspect pinning, stream/event
ordering, prefetch distance, buffer lifetime and page faults when relevant.
Check whether data is transferred early enough for the consumer and whether
prefetch competes with current compute.

Do not recommend removing offload without demonstrating memory feasibility.
Similarly, asynchronous copies require appropriate buffers and dependencies;
an asynchronous API does not prove useful overlap. A copy can be necessary while
an intermediate representation or duplicate copy is avoidable (L3).

## Specialized residual model

The X-Stage remote-store FA+A2A residual equations retained in
[methodology](../methodology.md#l5-distributed-communication) apply only when the
same mechanism, timing boundaries and independent calibration are justified.
Do not transfer their parameters to generic NCCL, another topology or sender-only
completion. Missing remote-consumer evidence blocks a distributed saturation claim.

## Handoff

Recommend only directions tied to the observed cause: readiness balance,
granularity/partition, compatible transport, fewer representations/copies or
residency/prefetch. Include memory feasibility, all-rank correctness and quality,
consumer-visible latency and loaded-service behavior where applicable.
