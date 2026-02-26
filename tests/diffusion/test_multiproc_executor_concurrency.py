# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.scheduler import Scheduler

pytestmark = [pytest.mark.diffusion]


# ───────────────────────────────────────────── helpers ─────────────────────


def _tagged_output(tag: str) -> DiffusionOutput:
    """Return a ``DiffusionOutput`` identifiable by its *error* field."""
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _mock_request(tag: str) -> Mock:
    """Return a mock ``OmniDiffusionRequest`` identifiable by *tag*."""
    req = Mock()
    req.request_ids = [tag]
    return req


def _make_scheduler():
    """Create a ``Scheduler`` whose *mq* / *result_mq* are backed by
    plain ``queue.Queue`` objects (thread-safe, no real IPC).

    Returns ``(scheduler, request_queue, result_queue)``.
    """
    sched = Scheduler()
    sched.num_workers = 1
    sched._lock = threading.Lock()

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()

    mock_mq = Mock()
    mock_mq.enqueue = req_q.put

    mock_rmq = Mock()
    mock_rmq.dequeue = lambda timeout=None: res_q.get(timeout=timeout if timeout else 10)

    sched.mq = mock_mq
    sched.result_mq = mock_rmq
    return sched, req_q, res_q


def _make_executor(scheduler):
    """Create a ``MultiprocDiffusionExecutor`` wired to *scheduler*
    without launching real worker processes.
    """
    od_cfg = Mock()
    od_cfg.num_gpus = 1
    with patch.object(MultiprocDiffusionExecutor, "_init_executor"):
        executor = MultiprocDiffusionExecutor(od_cfg)
    executor.scheduler = scheduler
    executor._closed = False
    executor._processes = []
    return executor


def _start_worker(req_q, res_q, count=2):
    """Simulate workers: read *count* requests from *req_q* and put
    tagged ``DiffusionOutput``s on *res_q* (FIFO order).
    """

    def _run():
        for _ in range(count):
            req = req_q.get(timeout=10)
            method = req.get("method", "")
            args = req.get("args", ())
            if method == "generate" and args and hasattr(args[0], "request_ids"):
                tag = f"result_for_{args[0].request_ids[0]}"
            elif args:
                tag = f"result_for_{args[0]}"
            else:
                tag = f"result_for_{method}"
            res_q.put(_tagged_output(tag))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _inject_interleave(scheduler):
    """Monkey-patch ``scheduler.mq.enqueue`` so that:

    * The thread named **thread_a** *blocks* after its enqueue until the
      thread named **thread_b** has finished entirely.
    * All other threads pass through unblocked.

    Returns ``(a_enqueued: Event, b_complete: Event)`` for wiring.
    """
    a_enqueued = threading.Event()
    b_complete = threading.Event()
    orig_enqueue = scheduler.mq.enqueue  # points to req_q.put

    def _controlled(item):
        orig_enqueue(item)
        if threading.current_thread().name == "thread_a":
            a_enqueued.set()  # tell B: "A has enqueued"
            b_complete.wait(5)  # block A until B finishes

    scheduler.mq.enqueue = _controlled
    return a_enqueued, b_complete


# ──────────────────── bug-reproduction: concurrent add_req ────────────────


class TestConcurrentAddReqBug:
    """Two concurrent ``Scheduler.add_req()`` calls swap results."""

    def test_results_are_correctly_routed(self):
        sched, req_q, res_q = _make_scheduler()
        a_enqueued, b_complete = _inject_interleave(sched)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, DiffusionOutput] = {}

        def _a():
            results["A"] = sched.add_req(_mock_request("A"))

        def _b():
            a_enqueued.wait(5)  # wait for A to enqueue
            results["B"] = sched.add_req(_mock_request("B"))
            b_complete.set()  # release A

        ta = threading.Thread(target=_a, name="thread_a")
        tb = threading.Thread(target=_b, name="thread_b")
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)
        wt.join(5)

        # With correct (locked) implementation both assertions hold.
        # The bug causes them to be swapped.
        assert results["A"].error == "result_for_A"
        assert results["B"].error == "result_for_B"


# ──────────────── bug-reproduction: concurrent collective_rpc ─────────────


class TestConcurrentCollectiveRpcBug:
    """Two concurrent ``collective_rpc()`` calls swap results."""

    def test_results_are_correctly_routed(self):
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)
        a_enqueued, b_complete = _inject_interleave(sched)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, object] = {}

        def _a():
            results["A"] = executor.collective_rpc(
                "ping",
                args=("call_A",),
                unique_reply_rank=0,
            )

        def _b():
            a_enqueued.wait(5)
            results["B"] = executor.collective_rpc(
                "ping",
                args=("call_B",),
                unique_reply_rank=0,
            )
            b_complete.set()

        ta = threading.Thread(target=_a, name="thread_a")
        tb = threading.Thread(target=_b, name="thread_b")
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)
        wt.join(5)

        assert results["A"].error == "result_for_call_A"
        assert results["B"].error == "result_for_call_B"


# ──────── bug-reproduction: add_req vs collective_rpc concurrently ────────


class TestConcurrentAddReqVsCollectiveRpcBug:
    """``add_req`` and ``collective_rpc`` running concurrently swap results."""

    def test_results_are_correctly_routed(self):
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)
        a_enqueued, b_complete = _inject_interleave(sched)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, object] = {}

        def _a():  # add_req path
            results["A"] = sched.add_req(_mock_request("A"))

        def _b():  # collective_rpc path
            a_enqueued.wait(5)
            results["B"] = executor.collective_rpc(
                "ping",
                args=("call_B",),
                unique_reply_rank=0,
            )
            b_complete.set()

        ta = threading.Thread(target=_a, name="thread_a")
        tb = threading.Thread(target=_b, name="thread_b")
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)
        wt.join(5)

        assert isinstance(results["A"], DiffusionOutput)
        assert results["A"].error == "result_for_A"
        assert results["B"].error == "result_for_call_B"


# ─────────────── backward-compatibility (serial) tests ────────────────────


class TestSerialOperations:
    """Verify correct behaviour for single-threaded (serial) usage.

    These tests must pass both **before** and **after** any concurrency fix
    is applied – they guard against regressions in the basic request path.
    """

    def test_serial_add_req_returns_correct_result(self):
        sched, req_q, res_q = _make_scheduler()
        wt = _start_worker(req_q, res_q, count=1)

        result = sched.add_req(_mock_request("X"))
        wt.join(5)

        assert isinstance(result, DiffusionOutput)
        assert result.error == "result_for_X"

    def test_serial_add_req_multiple_sequential(self):
        sched, req_q, res_q = _make_scheduler()
        wt = _start_worker(req_q, res_q, count=3)

        for tag in ("one", "two", "three"):
            out = sched.add_req(_mock_request(tag))
            assert out.error == f"result_for_{tag}"

        wt.join(5)

    def test_serial_collective_rpc_single_rank(self):
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)
        wt = _start_worker(req_q, res_q, count=1)

        result = executor.collective_rpc(
            "ping",
            args=("Y",),
            unique_reply_rank=0,
        )
        wt.join(5)

        assert result.error == "result_for_Y"

    def test_serial_collective_rpc_all_ranks(self):
        """``collective_rpc`` without *unique_reply_rank* collects
        ``num_gpus`` responses.
        """
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)
        executor.od_config.num_gpus = 2

        # Pre-populate two results (simulating two workers replying)
        res_q.put(_tagged_output("rank0"))
        res_q.put(_tagged_output("rank1"))

        results = executor.collective_rpc("ping", args=("multi",))

        assert len(results) == 2
        assert results[0].error == "rank0"
        assert results[1].error == "rank1"

    def test_serial_add_req_then_collective_rpc(self):
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)
        wt = _start_worker(req_q, res_q, count=2)

        gen_out = sched.add_req(_mock_request("gen"))
        rpc_out = executor.collective_rpc(
            "ping",
            args=("rpc",),
            unique_reply_rank=0,
        )
        wt.join(5)

        assert gen_out.error == "result_for_gen"
        assert rpc_out.error == "result_for_rpc"

    def test_serial_add_req_error_propagation(self):
        """``add_req`` should raise when the worker reports an error."""
        sched, _, res_q = _make_scheduler()
        # Put an error response directly
        res_q.put({"status": "error", "error": "boom"})

        with pytest.raises(RuntimeError, match="worker error"):
            sched.add_req(_mock_request("fail"))

    def test_serial_collective_rpc_error_propagation(self):
        """``collective_rpc`` should raise when the worker reports an error."""
        sched, _, res_q = _make_scheduler()
        executor = _make_executor(sched)
        res_q.put({"status": "error", "error": "kaboom"})

        with pytest.raises(RuntimeError, match="kaboom"):
            executor.collective_rpc("bad", unique_reply_rank=0)

    def test_collective_rpc_closed_executor_raises(self):
        sched, _, _ = _make_scheduler()
        executor = _make_executor(sched)
        executor._closed = True

        with pytest.raises(RuntimeError, match="closed"):
            executor.collective_rpc("anything")


# ─────────── timeout regression: RPC must not block on a stalled lock ─────


class TestCollectiveRpcTimeoutWhileLockHeld:
    """``collective_rpc(timeout=...)`` must honour its timeout even when
    another thread holds ``scheduler._lock`` indefinitely (e.g. a stalled
    ``add_req`` waiting on an unresponsive worker).
    """

    def test_rpc_times_out_when_lock_held_directly(self):
        """Simplest case: lock is manually held by another thread."""
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)

        stall_started = threading.Event()

        def _hold_lock():
            sched._lock.acquire()
            stall_started.set()
            # Hold the lock far longer than the RPC timeout.
            threading.Event().wait(30)
            sched._lock.release()

        stall_thread = threading.Thread(target=_hold_lock, daemon=True)
        stall_thread.start()
        stall_started.wait(5)

        # collective_rpc should raise TimeoutError, NOT block forever.
        with pytest.raises(TimeoutError):
            executor.collective_rpc("health", timeout=0.5)

    def test_rpc_times_out_when_add_req_stalled_on_worker(self):
        """Real-world scenario the bot flagged:

        ``add_req`` holds ``_lock`` while blocked on ``result_mq.dequeue()``
        because the worker never replies.  A concurrent
        ``collective_rpc(timeout=...)`` must still time out instead of
        hanging forever waiting for the lock.
        """
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)

        add_req_blocked = threading.Event()

        # Patch dequeue: signal once entered, then block indefinitely
        # (simulates a worker that never sends a result).
        orig_dequeue = sched.result_mq.dequeue

        def _hanging_dequeue(timeout=None):
            add_req_blocked.set()
            # Block forever — the worker is "hung".
            threading.Event().wait(30)
            return orig_dequeue(timeout=timeout)

        sched.result_mq.dequeue = _hanging_dequeue

        # Thread running add_req — acquires the lock, enqueues, then
        # blocks on dequeue forever (worker hang).
        def _stalled_add_req():
            try:
                sched.add_req(_mock_request("stalled"))
            except Exception:
                pass

        t = threading.Thread(target=_stalled_add_req, daemon=True)
        t.start()

        # Wait until add_req is truly inside the lock and blocking.
        add_req_blocked.wait(5)

        # collective_rpc should time out at lock acquisition, not hang.
        with pytest.raises(TimeoutError):
            executor.collective_rpc("health_check", timeout=0.5)

    def test_rpc_without_timeout_still_waits_for_lock(self):
        """When no timeout is given, ``collective_rpc`` should still wait
        for the lock (blocking) — existing behaviour preserved.
        """
        sched, req_q, res_q = _make_scheduler()
        executor = _make_executor(sched)

        lock_released = threading.Event()

        def _hold_and_release():
            sched._lock.acquire()
            # Hold for a short time then release.
            threading.Event().wait(0.3)
            sched._lock.release()
            lock_released.set()

        # Pre-populate a result so collective_rpc succeeds after lock.
        res_q.put(_tagged_output("ok"))

        t = threading.Thread(target=_hold_and_release, daemon=True)
        t.start()

        # No timeout → should block until lock is released, then succeed.
        result = executor.collective_rpc(
            "ping",
            args=("wait",),
            unique_reply_rank=0,
        )
        t.join(5)

        assert result.error == "ok"
