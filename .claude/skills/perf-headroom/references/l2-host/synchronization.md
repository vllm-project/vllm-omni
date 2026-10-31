# Synchronization: necessary dependency or avoidable round trip?

## Detection

Inspect long CUDA wait APIs, repeated scalar D2H, stream/event waits, host branches
on device results, and step boundaries that force completion. Search the executed
source for `.item()`, tensor truth conversion, host materialization, explicit
synchronization and shape-dependent output operations. These are leads, not proof:
behavior depends on device, API, pinning, stream and whether a result is consumed.

## Trace the dependency before recommending removal

1. Identify the produced value, its first consumer and the reason it must be ready.
2. Determine whether the CPU consumes its contents or merely forwards metadata.
3. Map the wait to producer stream/event and previous asynchronous work. A slow
   wait may simply reveal required computation that was submitted earlier.
4. Check the wait's scope: one event, one stream, all device work, remote rank,
   or process message. Preserve ownership/lifetime and error propagation.
5. Separate natural execution from synchronization introduced by a diagnostic timer.

## Patterns and recommendation branches

| Pattern | What must be established | Possible recommendation |
| --- | --- | --- |
| Per-step scalar readback | Next step need, timestep/control semantics and actual D2H/wait | Keep eligible state on device or move invariant host computation; no silent algorithm change |
| Device data converted for logging | Whether logging needs this value synchronously | Defer/sample diagnostics in a separately authorized change; quantify exposed delay |
| Broad synchronization | Actual consumer set and unrelated streams being waited for | Narrow dependency to the owning event if safe; do not delete required ordering |
| CPU copy before a required output | Payload completion boundary and async output support | Analyze overlap with later independent work; current request still needs its payload |
| Dynamic output shape | Whether shape discovery needs device contents | Investigate fixed-capacity representation only if consumers can preserve semantics |
| Graph-incompatible round trip | Actual graph break and external dependency | Propose capture-safe reformulation only after validity and output checks |

Not all copies can overlap: pinned/pageable memory, stream ordering, buffer lifetime
and hardware copy resources matter. The word `non_blocking` is not evidence that
the consumer safely overlaps useful work.

## Interpretation

For a 300 ms wait, split producer completion from any additional delay after
readiness. Do not report the entire wait as removable overhead. Even a genuinely
avoidable wait may expose another dependency once removed. Estimate using L6's
dependency scenario, and leave new resource contention uncertain.

## Output and validation proposal

For each finding state value/owner/consumer, why the wait exists, current scope,
exposure, legal alternative, counter-signal and acceptance checks. Verify request
isolation, stream lifetime, graph/eager behavior, errors/cancellation and numerical
quality where relevant. This guide authorizes diagnosis, not editing synchronization.
