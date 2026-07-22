#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the engine output-queue handoff.

Isolates the Orchestrator -> server handoff over the janus output queue (no
model or GPU required) to quantify the difference between two consumer
strategies:

* ``OLD``  -- ``sync_q.get_nowait()`` returning ``None`` on empty, then
  ``asyncio.sleep(poll_interval)`` before retrying (the pre-change behaviour,
  ``_FINAL_OUTPUT_IDLE_SLEEP_S = 1ms``).
* ``NEW``  -- ``await async_q.get()``, parking until a message arrives.

It mirrors production: a producer **thread** (the Orchestrator runs in its own
thread) emits messages to an **async consumer** (the server event loop). Two
things are measured:

1. Per-message wakeup latency, with messages arriving at a random phase of the
   poll cycle (the streaming regime, where token/chunk generation is far slower
   than queue ops so the consumer is almost always waiting).
2. Idle cost (wakeups/s and CPU) while no messages are flowing.

Usage::

    python benchmarks/engine/queue_handoff_latency.py
    python benchmarks/engine/queue_handoff_latency.py --poll-ms 1.0 --messages 3000 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import random
import statistics
import threading
import time

import janus


def _producer(q: janus.Queue, n: int, gap_max_s: float, emit_ts: list[float], ready: threading.Event) -> None:
    """Emit ``n`` messages from a background thread, like the Orchestrator."""
    ready.wait()
    for i in range(n):
        # A random inter-message gap models generation being far slower than
        # queue ops, so the consumer is genuinely waiting most of the time.
        time.sleep(random.uniform(0.0, gap_max_s))
        emit_ts.append(time.perf_counter())
        q.sync_q.put_nowait(i)


async def _consume_old(q: janus.Queue, n: int, poll_s: float, recv_ts: list[float]) -> None:
    got = 0
    while got < n:
        try:
            q.sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(poll_s)
            continue
        recv_ts.append(time.perf_counter())
        got += 1


async def _consume_new(q: janus.Queue, n: int, poll_s: float, recv_ts: list[float]) -> None:
    for _ in range(n):
        await q.async_q.get()
        recv_ts.append(time.perf_counter())


async def _measure_latency(consumer, *, n: int, poll_s: float, gap_max_s: float) -> list[float]:
    q: janus.Queue = janus.Queue()
    emit_ts: list[float] = []
    recv_ts: list[float] = []
    ready = threading.Event()
    th = threading.Thread(target=_producer, args=(q, n, gap_max_s, emit_ts, ready), daemon=True)
    th.start()
    task = asyncio.ensure_future(consumer(q, n, poll_s, recv_ts))
    ready.set()
    await task
    th.join()
    q.shutdown()
    return sorted((r - e) * 1e6 for e, r in zip(emit_ts, recv_ts))  # microseconds


async def _measure_idle(strategy: str, *, poll_s: float, secs: float) -> tuple[int, float]:
    q: janus.Queue = janus.Queue()
    wakeups = 0
    cpu0 = time.process_time()
    if strategy == "old":
        end = time.perf_counter() + secs
        while time.perf_counter() < end:
            try:
                q.sync_q.get_nowait()
            except queue.Empty:
                wakeups += 1
            await asyncio.sleep(poll_s)
    else:
        task = asyncio.ensure_future(q.async_q.get())  # parks once, never wakes
        wakeups = 1
        await asyncio.sleep(secs)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    cpu_ms = (time.process_time() - cpu0) * 1e3
    q.shutdown()
    return wakeups, cpu_ms


def _summarize(label: str, lat: list[float]) -> None:
    print(
        f"  {label}: mean={statistics.mean(lat):7.1f}us  p50={lat[len(lat) // 2]:7.1f}us  "
        f"p99={lat[int(len(lat) * 0.99)]:7.1f}us  max={lat[-1]:7.1f}us"
    )


async def _main_async(args: argparse.Namespace) -> None:
    poll_s = args.poll_ms / 1e3
    gap_max_s = args.gap_max_ms / 1e3
    print(
        f"poll interval = {args.poll_ms:.1f} ms, messages/run = {args.messages}, "
        f"runs = {args.runs}, max inter-message gap = {args.gap_max_ms:.1f} ms\n"
    )

    print("== per-message wakeup latency (producer thread -> async consumer) ==")
    for label, consumer in (("OLD poll ", _consume_old), ("NEW await", _consume_new)):
        agg: list[float] = []
        for _ in range(args.runs):
            agg.extend(await _measure_latency(consumer, n=args.messages, poll_s=poll_s, gap_max_s=gap_max_s))
        _summarize(label, sorted(agg))

    print("\n== idle cost (no messages flowing) ==")
    for label, strategy in (("OLD poll ", "old"), ("NEW await", "new")):
        wakeups, cpu_ms = await _measure_idle(strategy, poll_s=poll_s, secs=args.idle_secs)
        print(
            f"  {label}: wakeups={wakeups:6d} ({wakeups / args.idle_secs:7.1f}/s)  "
            f"cpu={cpu_ms:6.1f} ms over {args.idle_secs:.0f}s idle"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poll-ms", type=float, default=1.0, help="Old poll interval in ms (default: 1.0).")
    parser.add_argument("--messages", type=int, default=3000, help="Messages per latency run (default: 3000).")
    parser.add_argument("--runs", type=int, default=3, help="Latency runs to aggregate (default: 3).")
    parser.add_argument(
        "--gap-max-ms", type=float, default=2.0, help="Max random inter-message gap in ms (default: 2.0)."
    )
    parser.add_argument(
        "--idle-secs", type=float, default=2.0, help="Idle measurement window in seconds (default: 2.0)."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility (default: 0).")
    args = parser.parse_args()
    random.seed(args.seed)
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
