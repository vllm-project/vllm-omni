# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import zmq
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched import RequestScheduler
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


# ───────────────────────────────────────────── helpers ─────────────────────


def _tagged_output(tag: str) -> DiffusionOutput:
    """Return a ``DiffusionOutput`` identifiable by its *error* field."""
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _mock_request(tag: str):
    """Return a lightweight request object identifiable by *tag*."""
    return SimpleNamespace(
        request_ids=[tag],
        prompts=[f"prompt_{tag}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )


def _make_executor(num_gpus: int = 1):
    """Create a ``MultiprocDiffusionExecutor`` without launching workers.

    Returns ``(executor, request_queue, result_queue)``.
    """
    od_cfg = SimpleNamespace(num_gpus=num_gpus)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_executor", lambda self: None)
    executor = MultiprocDiffusionExecutor(od_cfg)
    monkeypatch.undo()

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()

    mock_broadcast_mq = SimpleNamespace(enqueue=req_q.put)

    mock_rmq = SimpleNamespace(dequeue=lambda timeout=None: res_q.get(timeout=timeout if timeout is not None else 10))

    executor._broadcast_mq = mock_broadcast_mq
    executor._result_mq = mock_rmq
    executor._closed = False
    executor._processes = []
    executor.is_failed = False
    executor._failure_callbacks = []
    return executor, req_q, res_q


def _make_engine(num_gpus: int = 1):
    """Create a lightweight ``DiffusionEngine`` wired to mocked executor."""
    executor, req_q, res_q = _make_executor(num_gpus)
    engine = DiffusionEngine.__new__(DiffusionEngine)
    sched = RequestScheduler()
    sched.initialize(SimpleNamespace())
    engine.scheduler = sched
    engine.executor = executor
    engine._rpc_lock = threading.RLock()
    engine.abort_queue = queue.Queue()
    engine.execute_fn = executor.execute_request
    return engine, executor, req_q, res_q


def _start_worker(req_q, res_q, count=2):
    """Simulate workers: read *count* requests from *req_q* and put
    tagged ``DiffusionOutput``s on *res_q* (FIFO order).
    """

    def _run():
        for _ in range(count):
            req = req_q.get(timeout=10)
            method = req.get("method", "")
            args = req.get("args", ())
            if method in {"generate", "execute_model"} and args and hasattr(args[0], "request_ids"):
                tag = f"result_for_{args[0].request_ids[0]}"
            elif args:
                tag = f"result_for_{args[0]}"
            else:
                tag = f"result_for_{method}"
            res_q.put(_tagged_output(tag))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _inject_interleave(executor):
    """Monkey-patch ``executor._broadcast_mq.enqueue`` so that:

    * The thread named **thread_a** *blocks* after its enqueue until the
      thread named **thread_b** has finished entirely.
    * All other threads pass through unblocked.

    Returns ``(a_enqueued: Event, b_complete: Event)`` for wiring.
    """
    a_enqueued = threading.Event()
    b_complete = threading.Event()
    orig_enqueue = executor._broadcast_mq.enqueue  # points to req_q.put

    def _controlled(item):
        orig_enqueue(item)
        if threading.current_thread().name == "thread_a":
            a_enqueued.set()  # tell B: "A has enqueued"
            b_complete.wait(5)  # block A until B finishes

    executor._broadcast_mq.enqueue = _controlled
    return a_enqueued, b_complete


# ───────────────── concurrent request execution ─────────────────


class TestConcurrentRequestExecution:
    """Concurrent request execution should not swap results."""

    def test_results_are_correctly_routed(self):
        engine, executor, req_q, res_q = _make_engine()
        a_enqueued, b_complete = _inject_interleave(executor)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, DiffusionOutput] = {}

        def _a():
            results["A"] = engine.add_req_and_wait_for_response(_mock_request("A"))

        def _b():
            a_enqueued.wait(5)  # wait for A to enqueue
            results["B"] = engine.add_req_and_wait_for_response(_mock_request("B"))
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


# ───────────────── concurrent collective RPC ─────────────────


class TestConcurrentCollectiveRpc:
    """Concurrent ``collective_rpc()`` calls should not swap results."""

    def test_results_are_correctly_routed(self):
        engine, executor, req_q, res_q = _make_engine()
        a_enqueued, b_complete = _inject_interleave(executor)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, object] = {}

        def _a():
            results["A"] = engine.collective_rpc(
                "ping",
                args=("call_A",),
                unique_reply_rank=0,
            )

        def _b():
            a_enqueued.wait(5)
            results["B"] = engine.collective_rpc(
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


# ──────────── concurrent request execution and collective RPC ────────────


class TestConcurrentRequestExecutionAndCollectiveRpc:
    """Request execution and ``collective_rpc()`` should not swap results."""

    def test_results_are_correctly_routed(self):
        engine, executor, req_q, res_q = _make_engine()
        a_enqueued, b_complete = _inject_interleave(executor)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, object] = {}

        def _a():  # request execution path
            results["A"] = engine.add_req_and_wait_for_response(_mock_request("A"))

        def _b():  # collective_rpc path
            a_enqueued.wait(5)
            results["B"] = engine.collective_rpc(
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


# ─────────────────────── serial operation coverage ───────────────────────


class TestSerialEngineOperations:
    """Verify correct behaviour for single-threaded (serial) usage.

    These tests must pass both **before** and **after** any concurrency fix
    is applied – they guard against regressions in the basic request path.
    """

    def test_serial_add_req_returns_correct_result(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=1)

        result = engine.add_req_and_wait_for_response(_mock_request("X"))
        wt.join(5)

        assert isinstance(result, DiffusionOutput)
        assert result.error == "result_for_X"

    def test_serial_add_req_multiple_sequential(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=3)

        for tag in ("one", "two", "three"):
            out = engine.add_req_and_wait_for_response(_mock_request(tag))
            assert out.error == f"result_for_{tag}"

        wt.join(5)

    def test_serial_collective_rpc_single_rank(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=1)

        result = engine.collective_rpc(
            "ping",
            args=("Y",),
            unique_reply_rank=0,
        )
        wt.join(5)

        assert result.error == "result_for_Y"

    def test_serial_collective_rpc_all_ranks(self):
        """``collective_rpc`` without *unique_reply_rank* returns a single
        response from rank 0 (only rank 0 has a result_mq).
        """
        engine, _, _, res_q = _make_engine(num_gpus=2)

        # Pre-populate one result (only rank 0 replies via result_mq)
        res_q.put(_tagged_output("rank0"))

        results = engine.collective_rpc("ping", args=("multi",))

        # Only 1 response expected since only rank 0 has result_mq
        assert len(results) == 1
        assert results[0].error == "rank0"

    def test_serial_add_req_then_collective_rpc(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=2)

        gen_out = engine.add_req_and_wait_for_response(_mock_request("gen"))
        rpc_out = engine.collective_rpc(
            "ping",
            args=("rpc",),
            unique_reply_rank=0,
        )
        wt.join(5)

        assert gen_out.error == "result_for_gen"
        assert rpc_out.error == "result_for_rpc"

    def test_serial_add_req_error_propagation(self):
        """``add_req`` should raise when the worker reports an error."""
        engine, _, _, res_q = _make_engine()
        # Put an error response directly
        res_q.put({"status": "error", "error": "boom"})

        out = engine.add_req_and_wait_for_response(_mock_request("fail"))

        assert isinstance(out, DiffusionOutput)
        assert out.error is not None
        assert "boom" in out.error

    def test_serial_collective_rpc_error_propagation(self):
        """``collective_rpc`` should raise when the worker reports an error."""
        engine, _, _, res_q = _make_engine()
        res_q.put({"status": "error", "error": "kaboom"})

        with pytest.raises(RuntimeError, match="kaboom"):
            engine.collective_rpc("bad", unique_reply_rank=0)

    def test_collective_rpc_closed_executor_raises(self):
        engine, executor, _, _ = _make_engine()
        executor._closed = True

        with pytest.raises(RuntimeError, match="closed"):
            engine.collective_rpc("anything")


# ─────────── timeout regression: RPC must not block on a stalled lock ─────


class TestCollectiveRpcTimeoutWhileLockHeld:
    """``collective_rpc(timeout=...)`` must honour its timeout even when
    another thread holds ``engine._rpc_lock`` indefinitely (e.g. request
    execution stalled on ``add_req_and_wait_for_response`` → ``execute_fn``
    → ``collective_rpc`` while blocked on an unresponsive worker).
    """

    def test_rpc_times_out_when_lock_held_directly(self):
        """Simplest case: lock is manually held by another thread."""
        engine, _, _, _ = _make_engine()

        stall_started = threading.Event()

        def _hold_lock():
            engine._rpc_lock.acquire()
            stall_started.set()
            # Hold the lock far longer than the RPC timeout.
            threading.Event().wait(30)
            engine._rpc_lock.release()

        stall_thread = threading.Thread(target=_hold_lock, daemon=True)
        stall_thread.start()
        stall_started.wait(5)

        # collective_rpc should raise TimeoutError, NOT block forever.
        with pytest.raises(TimeoutError):
            engine.collective_rpc("health", timeout=0.5)

    def test_rpc_times_out_when_request_execution_stalled_on_worker(self):
        """Real-world scenario the bot flagged:

        The scheduler/execute path holds ``_rpc_lock`` while blocked on
        ``executor._result_mq.dequeue()`` because the worker never replies.
        A concurrent ``collective_rpc(timeout=...)`` must still time out
        instead of hanging forever waiting for the lock.
        """
        engine, executor, _, _ = _make_engine()

        add_req_blocked = threading.Event()

        # Patch dequeue: signal once entered, then block indefinitely
        # (simulates a worker that never sends a result).
        orig_dequeue = executor._result_mq.dequeue

        def _hanging_dequeue(timeout=None):
            add_req_blocked.set()
            # Block forever — the worker is "hung".
            threading.Event().wait(30)
            return orig_dequeue(timeout=timeout)

        executor._result_mq.dequeue = _hanging_dequeue

        # Thread running request execution — acquires the lock, enqueues, then
        # blocks on dequeue forever (worker hang).
        def _stalled_request_execution():
            try:
                engine.add_req_and_wait_for_response(_mock_request("stalled"))
            except Exception:
                pass

        t = threading.Thread(target=_stalled_request_execution, daemon=True)
        t.start()

        # Wait until request execution is truly inside the lock and blocking.
        assert add_req_blocked.wait(5)

        # collective_rpc should time out at lock acquisition, not hang.
        with pytest.raises(TimeoutError):
            engine.collective_rpc("health_check", timeout=0.5)

    def test_rpc_without_timeout_still_waits_for_lock(self):
        """When no timeout is given, ``collective_rpc`` should still wait
        for the lock (blocking) — existing behaviour preserved.
        """
        engine, _, _, res_q = _make_engine()

        def _hold_and_release():
            engine._rpc_lock.acquire()
            # Hold for a short time then release.
            threading.Event().wait(0.3)
            engine._rpc_lock.release()

        # Pre-populate a result so collective_rpc succeeds after lock.
        res_q.put(_tagged_output("ok"))

        t = threading.Thread(target=_hold_and_release, daemon=True)
        t.start()

        # No timeout -> should block until lock is released, then succeed.
        result = engine.collective_rpc(
            "ping",
            args=("wait",),
            unique_reply_rank=0,
        )
        t.join(5)

        assert result.error == "ok"


# ───────── error handling: EngineDeadError propagation through layers ─────


class TestMultiprocExecutorRaisesEngineDeadError:
    """``collective_rpc`` raises ``EngineDeadError`` when the engine is failed."""

    def test_collective_rpc_raises_when_is_failed(self):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._closed = False
        executor._broadcast_mq = MagicMock()
        executor._result_mq = MagicMock()
        executor._result_mq.dequeue = MagicMock(side_effect=TimeoutError)
        executor.is_failed = True

        with pytest.raises(EngineDeadError):
            executor.collective_rpc(
                "generate",
                args=(MagicMock(),),
                unique_reply_rank=0,
                exec_all_ranks=True,
            )

    def test_collective_rpc_raises_mid_dequeue_when_is_failed(self):
        """Worker dies while we are polling the dequeue loop."""
        executor, _, res_q = _make_executor()

        call_count = 0
        orig_dequeue = executor._result_mq.dequeue

        def _dying_dequeue(timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                executor.is_failed = True
                raise TimeoutError
            return orig_dequeue(timeout=timeout)

        executor._result_mq.dequeue = _dying_dequeue

        with pytest.raises(EngineDeadError):
            executor.collective_rpc(
                "generate",
                args=(MagicMock(),),
                unique_reply_rank=0,
                exec_all_ranks=True,
            )


class TestDiffusionEngineDeadErrorPassthrough:
    """``DiffusionEngine.add_req_and_wait_for_response`` re-raises
    ``EngineDeadError`` from executor and wraps other errors."""

    def test_engine_dead_error_propagates(self):
        engine, executor, _, _ = _make_engine()
        engine.execute_fn = Mock(side_effect=EngineDeadError())

        with pytest.raises(EngineDeadError):
            engine.add_req_and_wait_for_response(_mock_request("dead"))

    def test_runtime_error_wrapped_in_output(self):
        engine, executor, _, _ = _make_engine()
        engine.execute_fn = Mock(side_effect=RuntimeError("gpu fault"))

        out = engine.add_req_and_wait_for_response(_mock_request("fault"))
        assert isinstance(out, DiffusionOutput)
        assert "gpu fault" in out.error


class TestStageDiffusionClientErrorPropagation:
    """Error surface behaviour of ``StageDiffusionClient``.

    Uses ``object.__new__`` to construct a client without spawning a real
    subprocess, then manually sets the fields needed for each test.
    """

    def _make_client(self, *, engine_dead=False, proc_alive=True):
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        client.stage_id = 0
        client.final_output = True
        client.final_output_type = "image"
        client.default_sampling_params = None
        client.custom_process_input_func = None
        client.engine_input_source = None

        client._output_queue = asyncio.Queue()
        client._rpc_results = {}
        client._pending_rpcs = set()
        client._tasks = {}
        client._shutting_down = False
        client._engine_dead = engine_dead
        client._owns_process = True
        client._proc = MagicMock(
            is_alive=MagicMock(return_value=proc_alive),
            exitcode=1,
        )
        client._request_socket = MagicMock()
        client._response_socket = MagicMock()
        client._encoder = MagicMock()
        client._decoder = MagicMock()

        return client

    @pytest.mark.asyncio
    async def test_add_request_raises_when_dead(self):
        client = self._make_client(engine_dead=True)

        with pytest.raises(EngineDeadError):
            await client.add_request_async("req-3", "test prompt", None)

    def test_check_health_raises_when_dead(self):
        client = self._make_client(engine_dead=True)

        with pytest.raises(EngineDeadError):
            client.check_health()

    def test_check_health_ok_when_alive(self):
        client = self._make_client()
        client.check_health()

    def test_get_output_raises_engine_dead_when_dead(self):
        """When ``_engine_dead`` is True and the output queue is empty,
        ``get_diffusion_output_nowait`` must raise ``EngineDeadError``."""
        client = self._make_client(engine_dead=True)
        # Simulate _drain_responses as a no-op (no ZMQ socket)
        client._response_socket.recv.side_effect = zmq.Again

        with pytest.raises(EngineDeadError):
            client.get_diffusion_output_nowait()

    def test_get_output_returns_none_when_alive_and_empty(self):
        """When the engine is alive and the queue is empty, return None."""
        client = self._make_client()
        client._response_socket.recv.side_effect = zmq.Again

        assert client.get_diffusion_output_nowait() is None

    def test_check_health_raises_when_proc_dead(self):
        """``check_health`` detects a dead subprocess via ``_proc.is_alive()``
        and raises ``EngineDeadError``, setting ``_engine_dead`` as a
        side effect."""
        client = self._make_client(proc_alive=False)

        with pytest.raises(EngineDeadError, match="not alive"):
            client.check_health()

        assert client._engine_dead is True

    def test_get_output_raises_when_proc_dead(self):
        """When the subprocess has died (non-signal exit) and the output
        queue is empty, ``get_diffusion_output_nowait`` must raise
        ``EngineDeadError`` with the exit code."""
        client = self._make_client(proc_alive=False)
        client._response_socket.recv.side_effect = zmq.Again

        with pytest.raises(EngineDeadError, match="exit code"):
            client.get_diffusion_output_nowait()

        assert client._engine_dead is True

    def test_get_output_returns_none_on_signal_death(self):
        """When the subprocess was killed by a signal (exit code > 128),
        ``get_diffusion_output_nowait`` returns ``None`` and sets
        ``_shutting_down`` instead of raising."""
        client = self._make_client(proc_alive=False)
        client._proc.exitcode = 137  # SIGKILL (128 + 9)
        client._response_socket.recv.side_effect = zmq.Again

        result = client.get_diffusion_output_nowait()

        assert result is None
        assert client._shutting_down is True
        assert client._engine_dead is True

    def test_initialize_client_requires_replica_id(self):
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        metadata = SimpleNamespace(
            stage_id=0,
            final_output=True,
            final_output_type="image",
            default_sampling_params=None,
            requires_multimodal_data=False,
            custom_process_input_func=None,
            engine_input_source=[],
        )

        with pytest.raises(AttributeError, match="replica_id"):
            client._initialize_client(
                metadata,
                "tcp://req",
                "tcp://resp",
                proc=None,
                batch_size=1,
            )


# ───────── monitor thread & death sentinel integration tests ─────────


def _poll_flag(get_flag, *, timeout=5.0, interval=0.05) -> bool:
    """Poll until ``get_flag()`` returns True or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if get_flag():
            return True
        time.sleep(interval)
    return False


def _make_short_lived_process() -> mp.Process:
    """Spawn a real subprocess that exits immediately.

    The process must be started with ``"fork"`` (or the platform default)
    so that it can use a plain ``lambda`` as its target — ``"spawn"`` would
    fail to pickle it.
    """
    ctx = mp.get_context("fork")
    p = ctx.Process(target=lambda: None, name="ShortLivedWorker-0")
    p.start()
    return p


class TestMultiprocExecutorWorkerMonitor:
    """Integration tests for ``start_worker_monitor``.

    Uses real short-lived subprocesses so that OS-level sentinel fd
    readiness is exercised end-to-end.
    """

    def test_worker_monitor_sets_is_failed_and_calls_callbacks_on_death(self):
        """When a worker process dies, the monitor thread must:
        1. Set ``is_failed = True``
        2. Call ``shutdown()`` (which sets ``_closed = True``)
        3. Invoke all registered failure callbacks
        """
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._closed = False
        executor.is_failed = False
        executor._failure_callbacks = []
        executor._broadcast_mq = None
        executor._result_mq = None
        executor.resources = None
        # Use a no-op so shutdown() doesn't crash on None resources.
        executor._finalizer = lambda: None

        proc = _make_short_lived_process()
        executor._processes = [proc]

        callback_called = threading.Event()
        executor.register_failure_callback(callback_called.set)

        executor.start_worker_monitor()

        # Wait for the process to exit and the monitor to react.
        proc.join(5)
        assert _poll_flag(lambda: executor.is_failed), "is_failed was not set"
        assert executor._closed, "shutdown() was not called"
        assert callback_called.wait(timeout=2), "failure callback was not invoked"

    def test_worker_monitor_noop_when_already_closed(self):
        """If ``_closed`` is already True when the process dies (orderly
        shutdown), the monitor must *not* set ``is_failed``."""
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._closed = True  # already shut down
        executor.is_failed = False
        executor._failure_callbacks = []
        executor._broadcast_mq = None
        executor._result_mq = None
        executor.resources = None
        executor._finalizer = lambda: None

        proc = _make_short_lived_process()
        executor._processes = [proc]

        executor.start_worker_monitor()
        proc.join(5)

        # Give the monitor thread a chance to run (it should early-return).
        time.sleep(0.3)
        assert not executor.is_failed, "is_failed should remain False on orderly shutdown"


class TestStageDiffusionClientProcMonitor:
    """Integration test for ``StageDiffusionClient._start_proc_monitor``.

    Uses a real short-lived subprocess to verify the sentinel-based
    detection pipeline.
    """

    def test_proc_monitor_sets_engine_dead_on_process_death(self):
        """When the subprocess dies, the monitor thread must set
        ``_engine_dead = True``."""
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        client.stage_id = 0
        client._shutting_down = False
        client._engine_dead = False

        proc = _make_short_lived_process()
        client._proc = proc

        client._start_proc_monitor()
        proc.join(5)

        assert _poll_flag(lambda: client._engine_dead), "_engine_dead was not set"


class TestDrainResponsesDeathSentinel:
    """Tests for death sentinel and error routing in
    ``StageDiffusionClient._drain_responses()``.
    """

    def _make_client(self):
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        client.stage_id = 0
        client._engine_dead = False
        client._shutting_down = False
        client._output_queue = asyncio.Queue()
        client._rpc_results = {}
        client._pending_rpcs = set()
        client._response_socket = MagicMock()
        client._decoder = MagicMock()
        return client

    def test_drain_responses_sets_engine_dead_on_death_sentinel(self):
        """When ``_drain_responses`` receives the ``DIFFUSION_PROC_DEAD``
        sentinel, it must set ``_engine_dead = True`` and stop draining
        (decoder is never called)."""
        client = self._make_client()

        # First recv returns the death sentinel, second would be a normal
        # message but should never be reached.
        client._response_socket.recv.side_effect = [
            StageDiffusionProc.DIFFUSION_PROC_DEAD,
            b"should-not-be-reached",
        ]

        client._drain_responses()

        assert client._engine_dead is True
        client._decoder.decode.assert_not_called()

    def test_drain_responses_routes_error_as_omni_request_output(self):
        """When ``_drain_responses`` receives a ``{"type": "error"}`` message
        with a ``request_id``, it must place an ``OmniRequestOutput`` with
        the error on ``_output_queue``."""
        client = self._make_client()

        error_msg = {
            "type": "error",
            "request_id": "req-fail",
            "error": "gpu fault",
        }
        # First recv returns the encoded error, second raises zmq.Again.
        client._response_socket.recv.side_effect = [b"encoded-error", zmq.Again]
        client._decoder.decode.return_value = error_msg

        client._drain_responses()

        assert not client._output_queue.empty()
        output = client._output_queue.get_nowait()
        assert isinstance(output, OmniRequestOutput)
        assert output.request_id == "req-fail"
        assert output.error == "gpu fault"
        assert output.finished is True
