# Engine microbenchmarks

Standalone microbenchmarks for engine-internal hot paths. No model or GPU
required — they isolate a single mechanism so the effect of a change is
measurable on its own.

## `queue_handoff_latency.py`

Quantifies the Orchestrator → server handoff over the janus output queue, the
path changed when the server's output reader moved from a 1 ms busy-poll to an
event-driven wakeup.

It mirrors production: a producer **thread** (the Orchestrator runs in its own
thread) emits messages to an **async consumer** (the server event loop), and
compares two consumer strategies:

- **OLD** — `sync_q.get_nowait()`, return `None` on empty, `asyncio.sleep(1ms)`,
  retry.
- **NEW** — the production `OutputQueueSink`: `arm_wakeup()`, drain, then park
  on the event the producer thread sets via `call_soon_threadsafe`, exactly as
  `OmniEngineBase.get_outputs_async` does.

```bash
python benchmarks/engine/queue_handoff_latency.py
python benchmarks/engine/queue_handoff_latency.py --poll-ms 1.0 --messages 3000 --runs 3
```

### Representative result

```text
== per-message wakeup latency (producer thread -> async consumer) ==
  OLD poll : mean=  575.7us  p50=  581.6us  p99= 1079.6us  max= 1284.7us
  NEW sink : mean=   96.8us  p50=   63.3us  p99=  337.8us  max=  678.0us

== idle cost (no messages flowing) ==
  OLD poll : wakeups=  1828 (  914.0/s)  cpu=  78.4 ms over 2s idle
  NEW sink : wakeups=     1 (    0.5/s)  cpu=   0.3 ms over 2s idle
```

### How to read it

- **Latency** — the old poll adds ~0.5 ms on average and up to ~1 ms per message
  (a message arriving at a uniformly random point in a 1 ms poll cycle waits
  half a cycle, hence p50 ≈ 0.5 ms, p99 ≈ the full interval). The sink cuts mean
  latency ~6× and p50 ~9×. The residual ~60–100 µs in the new path is the
  inherent cost of waking an asyncio task from another thread; polling cannot
  beat that.
- **Idle cost** — the old loop wakes ~900×/s and burns ~5% of a core per idle
  engine; the new one parks and costs almost nothing.

The latency win only materializes while the consumer is actually waiting — but
that is the normal streaming regime, where token/audio-chunk generation is far
slower than a queue op, so the consumer waits for essentially every output.
Numbers vary with machine and event-loop timer granularity; run it on the
target host for a local baseline.
