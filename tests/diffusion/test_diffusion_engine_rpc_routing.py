# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the busy-loop-based RPC routing in DiffusionEngine.

These tests guard the invariants established when ``collective_rpc`` was
moved off ``_rpc_lock`` mutual exclusion and onto a queue drained by the
engine's busy-loop thread:

* Only the busy-loop thread ever calls ``executor.collective_rpc`` after
  the loop starts.
* Concurrent ``engine.collective_rpc`` and per-request ``execute_fn``
  invocations never overlap on the executor.
* Results are routed back to the correct caller (sync, async, and
  per-request paths).
* Pending RPCs are failed cleanly on shutdown.
* The bootstrap path (busy loop not yet started) calls the executor
  directly so ``_dummy_run`` keeps working.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import fields as _dc_fields
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, _RpcTask
from vllm_omni.diffusion.sched import RequestScheduler
from vllm_omni.diffusion.sched.interface import SamplingParamsKey
from vllm_omni.diffusion.worker.utils import RunnerOutput

# Default values for every batch-key field, so SimpleNamespace-based
# sampling_params satisfy ``get_sampling_params_key``'s attribute lookups.
_SAMPLING_KEY_DEFAULTS = {f.name: f.default for f in _dc_fields(SamplingParamsKey)}

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


# ───────────────────────────────────────── helpers ─────────────────────────


class _ConcurrencyTrackingExecutor:
    """Fake executor that records every ``collective_rpc`` invocation and
    flags any overlap. Used to assert structural serialization.
    """

    def __init__(self, rpc_delay: float = 0.0):
        self._active = 0
        self._lock = threading.Lock()
        self.max_concurrent = 0
        self.calls: list[dict[str, Any]] = []
        self.thread_ids: set[int] = set()
        self.rpc_delay = rpc_delay
        self.is_failed = False
        self._closed = False

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
    ) -> Any:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.thread_ids.add(threading.get_ident())
            self.calls.append(
                {
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                    "unique_reply_rank": unique_reply_rank,
                    "thread": threading.get_ident(),
                }
            )
        try:
            if self.rpc_delay:
                time.sleep(self.rpc_delay)
            # Distinguish per-request execution from raw RPC by method name.
            if method in {"execute_model", "execute_stepwise", "generate"}:
                # args[0] is the request-like object for execute_model.
                req = args[0] if args else None
                tag = req.request_ids[0] if req is not None and hasattr(req, "request_ids") else "unknown"
                return DiffusionOutput(error=f"result_for_{tag}")
            tag = args[0] if args else method
            return DiffusionOutput(error=f"rpc_result_for_{tag}")
        finally:
            with self._lock:
                self._active -= 1

    def execute_request(self, scheduler_output) -> RunnerOutput:
        # Mimic the real MultiprocDiffusionExecutor.execute_request: it
        # forwards a single request through collective_rpc.
        new_req = scheduler_output.scheduled_new_reqs[0]
        result = self.collective_rpc(
            "execute_model",
            args=(new_req.req,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        return RunnerOutput(
            req_id=new_req.sched_req_id,
            step_index=None,
            finished=True,
            result=result,
        )

    def shutdown(self) -> None:
        self._closed = True


def _make_request(tag: str):
    return SimpleNamespace(
        request_ids=[tag],
        prompts=[f"prompt_{tag}"],
        sampling_params=SimpleNamespace(num_inference_steps=1, **_SAMPLING_KEY_DEFAULTS),
    )


def _make_engine_with_loop(
    loop: asyncio.AbstractEventLoop,
    rpc_delay: float = 0.0,
):
    """Construct a ``DiffusionEngine`` skeleton with a real busy loop.

    The engine is wired with a fake executor that asserts no concurrent
    calls and a real ``RequestScheduler``.
    """
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.executor = _ConcurrencyTrackingExecutor(rpc_delay=rpc_delay)

    sched = RequestScheduler()
    sched.initialize(SimpleNamespace(max_num_seqs=1))
    engine.scheduler = sched
    engine.step_execution = False
    engine.execute_fn = engine.executor.execute_request

    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._out_queue = {}
    engine.abort_queue = queue.Queue()
    engine._rpc_queue = queue.Queue()

    engine.main_loop = loop
    engine.stop_event = threading.Event()
    engine._loop_started = True
    engine.worker_thread = threading.Thread(target=engine._busy_loop, daemon=True)
    engine.worker_thread.start()
    return engine


def _stop_engine(engine: DiffusionEngine) -> None:
    with engine._cv:
        engine.stop_event.set()
        engine._cv.notify_all()
    # Race-proof shutdown for the test: drain any RPCs still queued and
    # fail them with the documented shutdown error before the busy loop
    # has a chance to pick them up after its current in-flight call
    # returns. The engine's own ``_fail_pending_rpcs`` then has nothing
    # left to do.
    while True:
        try:
            task = engine._rpc_queue.get_nowait()
        except queue.Empty:
            break
        if not task.future.done():
            task.future.set_exception(RuntimeError("DiffusionEngine is shutting down."))
    engine.worker_thread.join(timeout=5)
    assert not engine.worker_thread.is_alive(), "Busy loop thread did not stop"


# ─────────────────────── single-thread invariant ───────────────────────────


@pytest.mark.asyncio
async def test_executor_only_called_from_busy_loop_thread():
    """All executor calls — both per-request and raw RPC — must come from
    the busy-loop thread, never from a caller's thread."""
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop)
    busy_tid = engine.worker_thread.ident
    try:
        # Per-request path
        await engine.async_add_req_and_wait_for_response(_make_request("req1"))
        # Raw RPC path (sync from a worker thread)
        result = await asyncio.to_thread(engine.collective_rpc, "ping", args=("a",), unique_reply_rank=0)
        assert result.error == "rpc_result_for_a"
        # Raw RPC path (async)
        result_async = await engine.async_collective_rpc("ping", args=("b",), unique_reply_rank=0)
        assert result_async.error == "rpc_result_for_b"
    finally:
        _stop_engine(engine)

    # Every recorded call ran on the busy-loop thread.
    assert engine.executor.thread_ids == {busy_tid}, (
        f"executor.collective_rpc must run only on the busy-loop thread "
        f"(expected {{{busy_tid}}}, got {engine.executor.thread_ids})"
    )


@pytest.mark.asyncio
async def test_executor_calls_never_overlap_under_load():
    """Stress: many concurrent ``collective_rpc`` callers + a request must
    never produce overlapping executor calls."""
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop, rpc_delay=0.005)
    try:

        async def _rpc(i: int):
            return await engine.async_collective_rpc("ping", args=(f"x{i}",), unique_reply_rank=0)

        async def _request(i: int):
            return await engine.async_add_req_and_wait_for_response(_make_request(f"r{i}"))

        tasks = [_rpc(i) for i in range(20)] + [_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
    finally:
        _stop_engine(engine)

    assert engine.executor.max_concurrent == 1, (
        f"Detected concurrent executor calls (max_concurrent={engine.executor.max_concurrent})"
    )
    # All 25 calls should have a result.
    assert len(results) == 25
    assert all(r is not None for r in results)


# ─────────────────────────── result routing ────────────────────────────────


@pytest.mark.asyncio
async def test_collective_rpc_results_routed_to_correct_caller():
    """Under concurrent calls with distinct args, each caller must get its
    own result back — not another caller's."""
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop, rpc_delay=0.01)
    try:

        async def _call(tag: str):
            res = await engine.async_collective_rpc("ping", args=(tag,), unique_reply_rank=0)
            return tag, res

        tags = [f"t{i}" for i in range(15)]
        results = await asyncio.gather(*[_call(t) for t in tags])
    finally:
        _stop_engine(engine)

    for tag, res in results:
        assert res.error == f"rpc_result_for_{tag}", f"caller for {tag!r} received {res.error!r}"


@pytest.mark.asyncio
async def test_sync_collective_rpc_from_worker_thread():
    """Sync ``collective_rpc`` from non-event-loop threads still receives
    the correct result via the busy loop."""
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop)
    try:
        results: list[Any] = [None] * 8

        def _worker(idx: int):
            results[idx] = engine.collective_rpc("ping", args=(f"s{idx}",), unique_reply_rank=0)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
            assert not t.is_alive()
    finally:
        _stop_engine(engine)

    for i, r in enumerate(results):
        assert r is not None
        assert r.error == f"rpc_result_for_s{i}"


# ─────────────────────────── timeout behaviour ─────────────────────────────


@pytest.mark.asyncio
async def test_collective_rpc_times_out_when_busy_loop_busy():
    """If the busy loop is occupied, ``collective_rpc(timeout=...)`` must
    raise ``TimeoutError`` rather than block indefinitely."""
    loop = asyncio.get_running_loop()
    # 2-second per-RPC delay simulates a busy worker.
    engine = _make_engine_with_loop(loop, rpc_delay=2.0)
    try:
        # Kick off one slow RPC to occupy the busy loop.
        slow = asyncio.create_task(engine.async_collective_rpc("slow", args=("s",), unique_reply_rank=0))
        # Yield so the slow task is enqueued.
        await asyncio.sleep(0.05)

        with pytest.raises(TimeoutError):
            await asyncio.to_thread(
                engine.collective_rpc,
                "ping",
                args=("x",),
                unique_reply_rank=0,
                timeout=0.2,
            )

        # The slow call should still complete normally.
        result = await slow
        assert result.error == "rpc_result_for_s"
    finally:
        _stop_engine(engine)


@pytest.mark.asyncio
async def test_async_collective_rpc_times_out_when_busy_loop_busy():
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop, rpc_delay=2.0)
    try:
        slow = asyncio.create_task(engine.async_collective_rpc("slow", args=("s",), unique_reply_rank=0))
        await asyncio.sleep(0.05)

        with pytest.raises(TimeoutError):
            await engine.async_collective_rpc("ping", args=("x",), unique_reply_rank=0, timeout=0.2)

        result = await slow
        assert result.error == "rpc_result_for_s"
    finally:
        _stop_engine(engine)


# ─────────────────────────── shutdown handling ─────────────────────────────


@pytest.mark.asyncio
async def test_pending_rpcs_failed_on_shutdown():
    """Shutdown must fail any RPCs still queued so callers don't hang."""
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop, rpc_delay=0.5)
    try:
        # Start a slow RPC that occupies the busy loop, then queue more.
        in_flight = asyncio.create_task(engine.async_collective_rpc("slow", args=("s",), unique_reply_rank=0))
        await asyncio.sleep(0.05)

        pending = [
            asyncio.create_task(engine.async_collective_rpc("ping", args=(f"p{i}",), unique_reply_rank=0))
            for i in range(3)
        ]
        # Give them a moment to enqueue.
        await asyncio.sleep(0.05)
    finally:
        _stop_engine(engine)

    # The in-flight one finishes normally; pending ones should fail.
    assert (await in_flight).error == "rpc_result_for_s"
    for t in pending:
        with pytest.raises(RuntimeError, match="shutting down"):
            await t


# ─────────────────────────── bootstrap path ────────────────────────────────


def test_collective_rpc_before_loop_starts_calls_executor_directly():
    """When ``_loop_started`` is False (e.g. inside ``_dummy_run`` during
    ``__init__``), ``collective_rpc`` must call the executor synchronously
    on the caller's thread without enqueueing.
    """
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine._loop_started = False
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine.executor = _ConcurrencyTrackingExecutor()

    caller_tid = threading.get_ident()
    result = engine.collective_rpc("ping", args=("boot",), unique_reply_rank=0)

    assert result.error == "rpc_result_for_boot"
    # Critically: ran on the caller's thread, not a busy-loop thread.
    assert engine.executor.thread_ids == {caller_tid}


def test_collective_rpc_before_loop_starts_serializes_concurrent_callers():
    """Regression: between ``__init__`` returning and the first async
    request starting the busy loop, multiple threads may call
    ``collective_rpc`` concurrently. The pre-loop fast-path must
    serialize them so they cannot race on the shared executor MQ pair.
    """
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine._loop_started = False
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    # Non-trivial delay forces overlap if the lock is missing.
    engine.executor = _ConcurrencyTrackingExecutor(rpc_delay=0.02)

    n = 8
    results: list[Any] = [None] * n

    def _call(i: int) -> None:
        results[i] = engine.collective_rpc("ping", args=(f"x{i}",), unique_reply_rank=0)

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert engine.executor.max_concurrent == 1, "Pre-loop collective_rpc must serialize concurrent callers"
    for i, r in enumerate(results):
        assert r is not None and r.error == f"rpc_result_for_x{i}"


@pytest.mark.asyncio
async def test_cancelled_future_does_not_kill_busy_loop():
    """Regression: if a queued RPC future is cancelled while the executor
    call is in flight, ``_process_rpc_queue`` must not raise
    ``InvalidStateError`` when trying to set the result/exception. Doing
    so would crash the busy-loop thread and stall all later requests.
    """
    loop = asyncio.get_running_loop()
    # Slow executor so we can cancel mid-flight.
    engine = _make_engine_with_loop(loop, rpc_delay=0.3)
    try:
        # Submit an RPC and immediately cancel its future to simulate
        # a sync timeout / asyncio cancellation racing the worker.
        with pytest.raises(TimeoutError):
            await asyncio.to_thread(
                engine.collective_rpc,
                "slow",
                args=("cancelme",),
                unique_reply_rank=0,
                timeout=0.05,
            )
        # Give the busy loop time to finish the in-flight slow call and
        # attempt to set state on the cancelled future.
        await asyncio.sleep(0.5)

        # If the busy loop crashed, this follow-up RPC would hang /
        # never complete. Bound it with a timeout to fail fast.
        result = await asyncio.wait_for(
            engine.async_collective_rpc("ping", args=("after",), unique_reply_rank=0),
            timeout=3.0,
        )
        assert result.error == "rpc_result_for_after"
        assert engine.worker_thread.is_alive()
    finally:
        _stop_engine(engine)


# ─────────────────────────── _RpcTask basics ───────────────────────────────


def test_rpc_task_default_future_is_unique_per_instance():
    """Regression: ``_RpcTask.future`` must default to a *new* Future per
    instance, not a shared one (would cross-resolve all callers).
    """
    a = _RpcTask(method="m", args=(), kwargs=None, deadline=None, unique_reply_rank=0)
    b = _RpcTask(method="m", args=(), kwargs=None, deadline=None, unique_reply_rank=0)
    assert a.future is not b.future
    a.future.set_result("a")
    assert not b.future.done()


# ──────────────── busy-loop drains RPCs without scheduler work ─────────────


@pytest.mark.asyncio
async def test_busy_loop_handles_rpc_without_pending_requests():
    """The wait predicate must wake on RPC submissions even when the
    scheduler queue is empty (regression: previously the loop only woke
    on ``has_requests()``).
    """
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop)
    try:
        # No request has been added; only RPC.
        result = await asyncio.wait_for(
            engine.async_collective_rpc("ping", args=("alone",), unique_reply_rank=0),
            timeout=3.0,
        )
        assert result.error == "rpc_result_for_alone"
    finally:
        _stop_engine(engine)


# ──────────────── interleaved RPC + request integrity ──────────────────────


@pytest.mark.asyncio
async def test_rpc_and_request_results_do_not_swap():
    """A concurrent RPC and request execution must each receive their own
    result (regression for the original race the refactor fixed).
    """
    loop = asyncio.get_running_loop()
    engine = _make_engine_with_loop(loop, rpc_delay=0.02)
    try:
        rpc_task = asyncio.create_task(engine.async_collective_rpc("ping", args=("rpc1",), unique_reply_rank=0))
        req_task = asyncio.create_task(engine.async_add_req_and_wait_for_response(_make_request("req1")))
        rpc_res, req_res = await asyncio.gather(rpc_task, req_task)
    finally:
        _stop_engine(engine)

    assert rpc_res.error == "rpc_result_for_rpc1"
    assert req_res.error == "result_for_req1"
    # Belt and braces: never overlapped on the executor.
    assert engine.executor.max_concurrent == 1
