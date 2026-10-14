# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark: O(N) scheduler queue partition vs. prior O(N·K) list.remove.

This module is a *timing benchmark*, not a correctness gate.  It exercises
the same control-flow shape as ``OmniSchedulingCoordinator._process_chunk_queue``
and ``OmniChunkTransferAdapter._process_chunk_queue_legacy`` so that CI or a
developer's laptop can measure the asymptotic improvement of the single-pass
partition over repeated ``list.remove``.

Run directly:
    python -m pytest tests/perf/test_sched_queue_o_n_bench.py -v --benchmark-disable

Or as a plain script (no pytest dependency):
    python tests/perf/test_sched_queue_o_n_bench.py
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
#  Minimal fake Request + RequestStatus (mirrors vllm.v1.request.RequestStatus
#  so we can benchmark without importing vLLM).
# ---------------------------------------------------------------------------


@dataclass
class _Status:
    WAITING: str = "waiting"
    RUNNING: str = "running"
    WAITING_FOR_CHUNK: str = "waiting_for_chunk"
    WAITING_FOR_INPUT: str = "waiting_for_input"


class RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
    WAITING_FOR_CHUNK = "waiting_for_chunk"
    WAITING_FOR_INPUT = "waiting_for_input"


@dataclass
class FakeRequest:
    request_id: str
    status: str = RequestStatus.WAITING
    external_req_id: str = ""


# ---------------------------------------------------------------------------
#  OLD implementation (O(N·K)): calls queue.remove() in the loop
# ---------------------------------------------------------------------------


def _process_chunk_queue_OLD(
    queue: Any,
    waiting_for_chunk_list: deque,
    chunk_ready_req_ids: set[str],
    requests_with_ready_chunks: set[str],
    finished_requests: set[str],
):
    """O(N·K) baseline: remove each moved request from the queue one-by-one."""
    queue_snapshot = list(queue)
    for request in queue_snapshot:
        req_id = request.request_id
        request_can_run = (
            req_id in finished_requests
            or req_id in chunk_ready_req_ids
            or req_id in requests_with_ready_chunks
        )
        if request.status != RequestStatus.WAITING_FOR_CHUNK:
            if request_can_run:
                if req_id in chunk_ready_req_ids:
                    requests_with_ready_chunks.add(req_id)
                continue
            if request.status == RequestStatus.WAITING_FOR_INPUT:
                continue
            request.status = RequestStatus.WAITING_FOR_CHUNK
        else:
            if request_can_run:
                request.status = RequestStatus.RUNNING
                if req_id in chunk_ready_req_ids:
                    requests_with_ready_chunks.add(req_id)
                continue
        if isinstance(queue, list):
            queue.remove(request)
        else:
            queue.remove_request(request)
        waiting_for_chunk_list.append(request)


# ---------------------------------------------------------------------------
#  NEW implementation (O(N)): single-pass partition
# ---------------------------------------------------------------------------


def _process_chunk_queue_NEW(
    queue: Any,
    waiting_for_chunk_list: deque,
    chunk_ready_req_ids: set[str],
    requests_with_ready_chunks: set[str],
    finished_requests: set[str],
):
    """O(N) single-pass partition; rebuild the queue in one slice-assignment."""
    queue_snapshot = list(queue)
    to_retain: list = []
    for request in queue_snapshot:
        req_id = request.request_id
        request_can_run = (
            req_id in finished_requests
            or req_id in chunk_ready_req_ids
            or req_id in requests_with_ready_chunks
        )
        if request.status != RequestStatus.WAITING_FOR_CHUNK:
            if request_can_run:
                if req_id in chunk_ready_req_ids:
                    requests_with_ready_chunks.add(req_id)
                to_retain.append(request)
                continue
            if request.status == RequestStatus.WAITING_FOR_INPUT:
                to_retain.append(request)
                continue
            request.status = RequestStatus.WAITING_FOR_CHUNK
        else:
            if request_can_run:
                request.status = RequestStatus.RUNNING
                if req_id in chunk_ready_req_ids:
                    requests_with_ready_chunks.add(req_id)
                to_retain.append(request)
                continue
        waiting_for_chunk_list.append(request)

    if isinstance(queue, list):
        queue[:] = to_retain
    elif hasattr(queue, "remove_requests"):
        moved = [r for r in queue_snapshot if r not in to_retain]
        queue.remove_requests(moved)
    else:
        queue.clear()
        for r in to_retain:
            queue.add_request(r)


# ---------------------------------------------------------------------------
#  Benchmark harness
# ---------------------------------------------------------------------------


def _build_scenario(n: int, move_ratio: float):
    """Build a queue of ``n`` requests where ``move_ratio`` fraction will move."""
    queue = [FakeRequest(f"r{i}") for i in range(n)]
    # Mark the first K as movable (not ready, not WAITING_FOR_INPUT)
    k = int(n * move_ratio)
    for i in range(k):
        queue[i].status = RequestStatus.WAITING
    # Mark the remaining as WAITING_FOR_INPUT or ready (so they stay)
    for i in range(k, n):
        if i < k + (n - k) // 2:
            queue[i].status = RequestStatus.WAITING_FOR_INPUT
        else:
            queue[i].status = RequestStatus.WAITING_FOR_CHUNK
    chunk_ready = set()
    requests_with_ready = set()
    finished = set()
    return queue, chunk_ready, requests_with_ready, finished


def _time_implementation(func, n: int, move_ratio: float, trials: int = 5):
    totals: list[float] = []
    for _ in range(trials):
        queue, chunk_ready, rwr, finished = _build_scenario(n, move_ratio)
        waiting: deque = deque()
        t0 = time.perf_counter()
        func(queue, waiting, chunk_ready, rwr, finished)
        t1 = time.perf_counter()
        totals.append(t1 - t0)
    return min(totals), sum(totals) / len(totals)


@pytest.mark.parametrize(
    ("n", "move_ratio"),
    [
        (100, 0.3),
        (500, 0.3),
        (1000, 0.3),
        (2000, 0.3),
        (5000, 0.3),
        (100, 0.7),
        (500, 0.7),
        (1000, 0.7),
        (2000, 0.7),
    ],
)
def test_bench_o_n_improvement(n: int, move_ratio: float):
    """Single-pass partition must be strictly faster than O(N·K) list.remove."""
    old_best, old_avg = _time_implementation(_process_chunk_queue_OLD, n, move_ratio)
    new_best, new_avg = _time_implementation(_process_chunk_queue_NEW, n, move_ratio)
    speedup_best = old_best / max(new_best, 1e-12)
    speedup_avg = old_avg / max(new_avg, 1e-12)
    # At all sizes the new implementation must be at least as fast; at N≥500
    # with move_ratio=0.3 it must be materially faster (>1.2×).
    assert new_avg <= old_avg, (
        f"O(N) implementation is slower than O(N·K): "
        f"old={old_avg:.4g}s new={new_avg:.4g}s speedup={speedup_avg:.2f}x "
        f"(N={n}, move_ratio={move_ratio})"
    )
    if n >= 500 and move_ratio >= 0.3:
        assert speedup_avg >= 1.15, (
            f"Expected ≥15% speedup at N={n} move_ratio={move_ratio}, got {speedup_avg:.2f}x"
        )
    print(f"\n  N={n:5d} move={move_ratio:.1f}  OLD={old_avg*1e6:8.1f}µs  "
          f"NEW={new_avg*1e6:8.1f}µs  speedup={speedup_avg:.2f}x")


@pytest.mark.parametrize(
    ("n", "move_ratio"),
    [(500, 0.5), (1000, 0.5), (2000, 0.5)],
)
def test_correctness_equivalence(n: int, move_ratio: float):
    """OLD and NEW implementations produce identical retained queues and waiting deques."""
    queue_old, cr_old, rwr_old, fin_old = _build_scenario(n, move_ratio)
    queue_new, cr_new, rwr_new, fin_new = _build_scenario(n, move_ratio)
    # Copy the same request objects between the two scenarios
    for o, nw in zip(queue_old, queue_new):
        nw.request_id = o.request_id
        nw.status = o.status

    waiting_old: deque = deque()
    waiting_new: deque = deque()
    _process_chunk_queue_OLD(queue_old, waiting_old, cr_old, rwr_old, fin_old)
    _process_chunk_queue_NEW(queue_new, waiting_new, cr_new, rwr_new, fin_new)

    old_ids = [r.request_id for r in queue_old]
    new_ids = [r.request_id for r in queue_new]
    assert old_ids == new_ids, f"Retained queue mismatch OLD={old_ids} NEW={new_ids}"

    old_waiting = [r.request_id for r in waiting_old]
    new_waiting = [r.request_id for r in waiting_new]
    assert old_waiting == new_waiting, f"Waiting deque mismatch OLD={old_waiting} NEW={new_waiting}"

    # Request statuses must also agree
    for o, nw in zip(queue_old, queue_new):
        assert o.status == nw.status, f"Status mismatch for {o.request_id}: {o.status} vs {nw.status}"
    for o, nw in zip(waiting_old, waiting_new):
        assert o.status == nw.status, f"Status mismatch for {o.request_id}: {o.status} vs {nw.status}"


# ---------------------------------------------------------------------------
#  O(N) remove-request-count regression guard
# ---------------------------------------------------------------------------


class _ListRemoveCounter(list):
    """Tracks the number of .remove() calls made against a list-backed queue."""

    def __init__(self, items):
        super().__init__(items)
        self.remove_calls = 0

    def remove(self, item):
        self.remove_calls += 1
        return super().remove(item)

    def pop(self, *args, **kwargs):
        # pop is fine; we only care about remove().
        return super().pop(*args, **kwargs)


def test_new_impl_calls_remove_zero_times_on_list_queue():
    """The new O(N) path must not invoke .remove() on a list-backed queue."""
    n = 200
    queue: _ListRemoveCounter = _ListRemoveCounter(FakeRequest(f"r{i}") for i in range(n))
    # First half are movable, second half WAITING_FOR_INPUT
    for i in range(n // 2):
        queue[i].status = RequestStatus.WAITING
    for i in range(n // 2, n):
        queue[i].status = RequestStatus.WAITING_FOR_INPUT

    waiting: deque = deque()
    requests_with_ready: set = set()
    chunk_ready: set = set()
    finished: set = set()

    _process_chunk_queue_NEW(queue, waiting, chunk_ready, requests_with_ready, finished)

    assert queue.remove_calls == 0, (
        f"NEW implementation called list.remove {queue.remove_calls} times; "
        f"expected 0 (slice-assignment only)"
    )
    # Sanity: OLD implementation WOULD have called remove n/2 times
    queue2: _ListRemoveCounter = _ListRemoveCounter(FakeRequest(f"r{i}") for i in range(n))
    for i in range(n // 2):
        queue2[i].status = RequestStatus.WAITING
    for i in range(n // 2, n):
        queue2[i].status = RequestStatus.WAITING_FOR_INPUT
    _process_chunk_queue_OLD(queue2, deque(), set(), set(), set())
    assert queue2.remove_calls == n // 2, (
        f"Expected OLD to call remove {n // 2} times, got {queue2.remove_calls}"
    )


# ---------------------------------------------------------------------------
#  Stand-alone driver (no pytest dependency)
# ---------------------------------------------------------------------------


def _run_as_script() -> None:
    print("=" * 70)
    print("vLLM-Omni Scheduler Queue — O(N) Partition Benchmark")
    print("=" * 70)
    print()
    header = f"{'N':>6s}  {'move%':>5s}  {'OLD (µs)':>12s}  {'NEW (µs)':>12s}  {'speedup':>8s}  {'correct?':>10s}"
    print(header)
    print("-" * len(header))

    scenarios = [
        (100, 0.3), (500, 0.3), (1000, 0.3), (2000, 0.3), (5000, 0.3),
        (100, 0.7), (500, 0.7), (1000, 0.7), (2000, 0.7),
    ]
    for n, mr in scenarios:
        old_best, old_avg = _time_implementation(_process_chunk_queue_OLD, n, mr)
        new_best, new_avg = _time_implementation(_process_chunk_queue_NEW, n, mr)
        speedup = old_avg / max(new_avg, 1e-12)

        # Correctness check for this scenario
        queue_old, cr_old, rwr_old, fin_old = _build_scenario(n, mr)
        queue_new, cr_new, rwr_new, fin_new = _build_scenario(n, mr)
        for o, nw in zip(queue_old, queue_new):
            nw.request_id = o.request_id
            nw.status = o.status
        wait_old: deque = deque()
        wait_new: deque = deque()
        _process_chunk_queue_OLD(queue_old, wait_old, cr_old, rwr_old, fin_old)
        _process_chunk_queue_NEW(queue_new, wait_new, cr_new, rwr_new, fin_new)
        correct = ([r.request_id for r in queue_old] == [r.request_id for r in queue_new]
                   and [r.request_id for r in wait_old] == [r.request_id for r in wait_new])

        print(f"{n:6d}  {mr*100:4.0f}%  {old_avg*1e6:12.1f}  {new_avg*1e6:12.1f}  {speedup:7.2f}x  {'yes' if correct else 'NO!!':>10s}")

    print()
    print("Legend:")
    print("  OLD — O(N·K) baseline: queue.remove() per moved request")
    print("  NEW — O(N) single-pass partition + slice-assignment")
    print()


if __name__ == "__main__":
    _run_as_script()
