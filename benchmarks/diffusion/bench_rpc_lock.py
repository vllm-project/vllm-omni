#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prospective benchmark: _rpc_lock scope impact on collective_rpc latency.

Simulates two locking strategies and measures how long a collective_rpc call
is blocked while a synthetic execute_fn is running. Pure threading.RLock
contention; no engine state. Use as the headline number when proposing future
narrowing of DiffusionEngine._rpc_lock.

Usage:
    python benchmarks/diffusion/bench_rpc_lock.py [--exec-time 2.0]
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time


def bench_wide_lock(exec_time: float, trials: int = 5) -> list[float]:
    """Simulate the current pattern: single lock around the entire method."""
    lock = threading.RLock()
    rpc_wait_times: list[float] = []

    def simulate_request():
        with lock:
            # scheduler.add_request + schedule
            time.sleep(0.001)
            # execute_fn (GPU work) — lock held!
            time.sleep(exec_time)
            # scheduler.update_from_output
            time.sleep(0.001)

    def measure_rpc():
        time.sleep(0.05)  # let request thread grab lock first
        start = time.perf_counter()
        with lock:
            pass  # collective_rpc work
        rpc_wait_times.append((time.perf_counter() - start) * 1000)

    for _ in range(trials):
        t_req = threading.Thread(target=simulate_request)
        t_rpc = threading.Thread(target=measure_rpc)
        t_req.start()
        t_rpc.start()
        t_req.join()
        t_rpc.join()

    return rpc_wait_times


def bench_narrow_lock(exec_time: float, trials: int = 5) -> list[float]:
    """Simulate a hypothetical narrowed pattern: lock released during execute_fn."""
    lock = threading.RLock()
    rpc_wait_times: list[float] = []

    def simulate_request():
        with lock:
            time.sleep(0.001)  # scheduler.add_request
        with lock:
            time.sleep(0.001)  # scheduler.schedule
        # execute_fn — lock RELEASED
        time.sleep(exec_time)
        with lock:
            time.sleep(0.001)  # scheduler.update_from_output

    def measure_rpc():
        time.sleep(0.05)  # let request thread start first
        start = time.perf_counter()
        with lock:
            pass  # collective_rpc work
        rpc_wait_times.append((time.perf_counter() - start) * 1000)

    for _ in range(trials):
        t_req = threading.Thread(target=simulate_request)
        t_rpc = threading.Thread(target=measure_rpc)
        t_req.start()
        t_rpc.start()
        t_req.join()
        t_rpc.join()

    return rpc_wait_times


def main():
    parser = argparse.ArgumentParser(description="Benchmark _rpc_lock scope")
    parser.add_argument("--exec-time", type=float, default=2.0, help="Simulated execute_fn duration in seconds")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per strategy")
    args = parser.parse_args()

    print(f"Simulated execute_fn duration: {args.exec_time:.1f}s, trials: {args.trials}\n")

    wide_times = bench_wide_lock(args.exec_time, args.trials)
    narrow_times = bench_narrow_lock(args.exec_time, args.trials)

    wide_mean = statistics.mean(wide_times)
    narrow_mean = statistics.mean(narrow_times)

    print("RPC lock acquisition latency (ms):")
    print(
        f"  Wide lock (current): mean={wide_mean:8.2f}  median={statistics.median(wide_times):8.2f}  "
        f"min={min(wide_times):8.2f}  max={max(wide_times):8.2f}"
    )
    print(
        f"  Narrow lock (proposed): mean={narrow_mean:8.2f}  median={statistics.median(narrow_times):8.2f}  "
        f"min={min(narrow_times):8.2f}  max={max(narrow_times):8.2f}"
    )
    print(f"\n  Speedup if narrowing is safe: {wide_mean / narrow_mean:.0f}x faster RPC response")


if __name__ == "__main__":
    main()
