# Engine microbenchmarks

Standalone microbenchmarks for engine-internal hot paths. No model or GPU
required — they isolate a single mechanism so the effect of a change is
measurable on its own.

## `queue_handoff_latency.py`

Quantifies the Orchestrator → server handoff over the janus output queue, the
path changed when the server's output reader moved from a 1 ms busy-poll to an
event-driven `await`.

It mirrors production: a producer **thread** (the Orchestrator runs in its own
thread) emits messages to an **async consumer** (the server event loop), and
compares two consumer strategies:

- **OLD** — `sync_q.get_nowait()`, return `None` on empty, `asyncio.sleep(1ms)`,
  retry.
- **NEW** — `await async_q.get()`, park until a message arrives.

```bash
python benchmarks/engine/queue_handoff_latency.py
python benchmarks/engine/queue_handoff_latency.py --poll-ms 1.0 --messages 3000 --runs 3
```

### Representative result

```
== per-message wakeup latency (producer thread -> async consumer) ==
  OLD poll : mean=  573.8us  p50=  574.7us  p99= 1103.6us  max= 1280.8us
  NEW await: mean=  144.7us  p50=   96.3us  p99=  398.3us  max=  585.1us

== idle cost (no messages flowing) ==
  OLD poll : wakeups=  1795 (  897.5/s)  cpu= 111.6 ms over 2s idle
  NEW await: wakeups=     1 (    0.5/s)  cpu=   0.4 ms over 2s idle
```

### How to read it

- **Latency** — the old poll adds ~0.5 ms on average and up to ~1 ms per message
  (a message arriving at a uniformly random point in a 1 ms poll cycle waits
  half a cycle, hence p50 ≈ 0.5 ms, p99 ≈ the full interval). The new path cuts
  mean latency ~4× and p50 ~6×. The residual ~100 µs in the new path is the
  inherent cost of waking an asyncio task from another thread; polling cannot
  beat that.
- **Idle cost** — the old loop wakes ~900×/s and burns ~5% of a core per idle
  engine; the new one parks and costs almost nothing.

The latency win only materializes while the consumer is actually waiting — but
that is the normal streaming regime, where token/audio-chunk generation is far
slower than a queue op, so the consumer waits for essentially every output.
Numbers vary with machine and event-loop timer granularity; run it on the
target host for a local baseline.
