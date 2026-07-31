"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies,
subsequent attempts to collect output raise RuntimeError.
"""

import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.async_engine_utils import weak_shutdown_async_omni_engine
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import (
    CollectiveRPCResultMessage,
    ErrorMessage,
    OutputMessage,
)
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient
from vllm_omni.experimental.fullduplex.engine.duplex_control_client import (
    DuplexControlClient,
    DuplexControlRequestError,
)
from vllm_omni.experimental.fullduplex.engine.messages import (
    AppendDuplexInputMessage,
    DuplexControlResultMessage,
    DuplexFence,
    SignalDuplexTurnMessage,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(output_queue, mocker: MockerFixture, *, thread_alive: bool = True) -> AsyncOmniEngine:
    """Create an AsyncOmniEngine bypassing __init__."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = output_queue
    engine.orchestrator_thread = mocker.MagicMock(
        is_alive=mocker.MagicMock(return_value=thread_alive),
    )
    return engine


def test_weak_shutdown_closes_rpc_router_before_joining_orchestrator(mocker: MockerFixture):
    request_queue = mocker.MagicMock()
    output_queue = mocker.MagicMock()
    rpc_output_queue = mocker.MagicMock()
    router = mocker.MagicMock()
    orchestrator_thread = mocker.MagicMock()
    orchestrator_thread.is_alive.return_value = True

    def assert_router_closed_before_join(*args, **kwargs):
        router.close.assert_called_once_with()

    orchestrator_thread.join.side_effect = assert_router_closed_before_join

    weak_shutdown_async_omni_engine(
        orchestrator_thread,
        request_queue,
        output_queue,
        rpc_output_queue,
        router,
    )

    request_queue.sync_q.put.assert_called_once()
    assert request_queue.sync_q.put.call_args.kwargs["timeout"] > 0
    orchestrator_thread.join.assert_called_once()
    assert orchestrator_thread.join.call_args.kwargs["timeout"] > 0


def test_try_get_output_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Draining remaining results then hitting an empty queue with a dead
    orchestrator must raise RuntimeError so callers know the pipeline is gone."""
    mock_queue = mocker.MagicMock()
    # First call succeeds; second call finds the queue empty.
    mock_queue.sync_q.get.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, mocker, thread_alive=True)

    # Collect the one buffered result.
    assert engine.try_get_output().request_id == "r1"

    # Orchestrator thread crashes between polls.
    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output()


@pytest.mark.asyncio
async def test_try_get_output_async_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Same scenario as above but for the async variant."""
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        )
    )

    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker, thread_alive=True)

    assert (await engine.try_get_output_async()).request_id == "r1"

    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        await engine.try_get_output_async()


def test_fatal_error_message_surfaces_through_try_get_output(mocker: MockerFixture):
    """When the orchestrator thread crashes, it enqueues a fatal error message.

    ``try_get_output`` must return this message so the caller
    (``OmniBase._handle_output_message``) can detect the fatal flag.
    """
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get.return_value = fatal_msg

    engine = _make_engine(mock_queue, mocker, thread_alive=False)

    msg = engine.try_get_output()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True
    assert "crashed" in msg.error


@pytest.mark.asyncio
async def test_fatal_error_message_surfaces_through_try_get_output_async(mocker: MockerFixture):
    """Async variant of the fatal error message test."""
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    raw_queue = queue.Queue()
    raw_queue.put_nowait(fatal_msg)

    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker, thread_alive=False)

    msg = await engine.try_get_output_async()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True


def test_output_remains_on_shared_output_path(mocker: MockerFixture):
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="legacy-duplex-request",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="legacy-duplex-request"),
            finished=False,
        )
    )
    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker)

    output = engine.try_get_output(timeout=0.01)

    assert output.request_id == "legacy-duplex-request"


def test_duplex_control_client_uses_engine_owned_correlated_transport():
    engine = object.__new__(AsyncOmniEngine)
    engine._duplex_control_client = None
    engine._correlated_rpc_client = CorrelatedRpcClient(queue.Queue(), queue.Queue())

    try:
        client = engine._get_duplex_control_client()

        assert isinstance(client, DuplexControlClient)
        assert client._transport is engine._correlated_rpc_client
        assert not hasattr(engine, "_submit_rpc_and_wait")
    finally:
        engine._correlated_rpc_client.close()


def test_duplex_control_api_requires_explicit_fence():
    engine = object.__new__(AsyncOmniEngine)

    with pytest.raises(TypeError, match="fence"):
        engine.open_duplex_session("sid")
    with pytest.raises(TypeError, match="fence"):
        engine.append_duplex_input("sid", mode="append_audio_chunk", payload={})
    with pytest.raises(TypeError, match="fence"):
        engine.signal_duplex_turn("sid", event="turn.end")
    with pytest.raises(TypeError, match="fence"):
        engine.close_duplex_session("sid")


def test_open_duplex_session_waits_for_control_ack(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    control_result = DuplexControlResultMessage(
        control_id="ctrl-1",
        fence=DuplexFence("sid"),
        operation="open",
        session_id="sid",
        ok=False,
        stage_results=[{"stage_id": 0, "replica_id": 0, "result": {"supported": False}}],
        unsupported_count=1,
        error_count=0,
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    engine._duplex_control_client = None
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="ctrl-1"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(engine.open_duplex_session, "sid", fence=DuplexFence("sid"), timeout=1)
        msg = request_q.get(timeout=1)
        assert msg.type == "open_duplex_session"
        assert msg.control_id == "ctrl-1"
        rpc_q.put(control_result)
        with pytest.raises(DuplexControlRequestError) as exc_info:
            pending.result(timeout=1)

    assert exc_info.value.result["unsupported_count"] == 1
    assert exc_info.value.result["stage_results"][0]["result"]["supported"] is False
    engine._correlated_rpc_client.close()


def test_signal_duplex_turn_message_carries_next_fence(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    cancelled_fence = DuplexFence("sid-cancel", epoch=4, turn_id=2)
    next_fence = DuplexFence("sid-cancel", epoch=5, turn_id=2)
    control_result = DuplexControlResultMessage(
        control_id="ctrl-signal",
        fence=cancelled_fence,
        operation="signal",
        session_id="sid-cancel",
        ok=True,
        stage_results=[],
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    engine._duplex_control_client = None
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="ctrl-signal"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            engine.signal_duplex_turn,
            "sid-cancel",
            event="input.cancel",
            fence=cancelled_fence,
            next_fence=next_fence,
            session_config={"temperature": 0.0},
            timeout=1,
        )
        msg = request_q.get(timeout=1)
        assert isinstance(msg, SignalDuplexTurnMessage)
        assert msg.fence == cancelled_fence
        assert msg.next_fence == next_fence
        assert msg.session_config == {"temperature": 0.0}
        rpc_q.put(control_result)
        assert pending.result(timeout=1)["ok"] is True

    engine._correlated_rpc_client.close()


@pytest.mark.asyncio
async def test_cancelled_async_append_does_not_stop_executor_callable(mocker: MockerFixture):
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    engine = object.__new__(AsyncOmniEngine)

    def blocking_append(*args, **kwargs):
        del args, kwargs
        entered.set()
        release.wait(timeout=1)
        completed.set()
        return {}

    mocker.patch.object(engine, "append_duplex_input", side_effect=blocking_append)
    task = asyncio.create_task(
        engine.append_duplex_input_async(
            "sid-executor-cancel",
            mode="append_audio_chunk",
            payload=b"audio",
            fence=DuplexFence("sid-executor-cancel"),
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()

    assert await asyncio.to_thread(completed.wait, 1)


def test_duplex_controls_route_out_of_order_without_global_lock():
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    engine._duplex_control_client = None
    first_fence = DuplexFence("sid", epoch=1)
    second_fence = DuplexFence("sid", epoch=2)
    first = AppendDuplexInputMessage(
        control_id="first",
        fence=first_fence,
        session_id="sid",
        expected_epoch=1,
        mode="audio",
        payload=b"first",
    )
    second = AppendDuplexInputMessage(
        control_id="second",
        fence=second_fence,
        session_id="sid",
        expected_epoch=2,
        mode="audio",
        payload=b"second",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        client = engine._get_duplex_control_client()
        first_result = executor.submit(client.execute, first, timeout=1)
        second_result = executor.submit(client.execute, second, timeout=1)
        submitted = {request_q.get(timeout=1).control_id, request_q.get(timeout=1).control_id}
        assert submitted == {"first", "second"}
        rpc_q.put(
            DuplexControlResultMessage(
                control_id="second",
                fence=second_fence,
                operation="append",
                session_id="sid",
                ok=True,
                stage_results=[],
            )
        )
        rpc_q.put(
            DuplexControlResultMessage(
                control_id="first",
                fence=first_fence,
                operation="append",
                session_id="sid",
                ok=True,
                stage_results=[],
            )
        )

        assert first_result.result(timeout=1)["fence"] == first_fence
        assert second_result.result(timeout=1)["fence"] == second_fence

    assert not hasattr(engine, "_rpc_lock")
    engine._correlated_rpc_client.close()


def test_terminal_rpc_router_rejects_without_enqueuing_new_control():
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    engine._duplex_control_client = None
    pending_message = AppendDuplexInputMessage(
        control_id="pending",
        fence=DuplexFence("sid"),
        session_id="sid",
        mode="audio",
        payload=b"audio",
    )
    message = AppendDuplexInputMessage(
        control_id="after-fatal",
        fence=DuplexFence("sid"),
        session_id="sid",
        mode="audio",
        payload=b"audio",
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(engine._get_duplex_control_client().execute, pending_message, timeout=1)
            assert request_q.get(timeout=1).control_id == "pending"
            rpc_q.put(ErrorMessage(error="orchestrator failed", fatal=True))
            with pytest.raises(RuntimeError, match="orchestrator failed"):
                pending.result(timeout=1)

        with pytest.raises(RuntimeError, match="orchestrator failed"):
            engine._get_duplex_control_client().execute(message, timeout=1)

        assert request_q.empty()
    finally:
        engine._correlated_rpc_client.close()


def test_collective_and_duplex_results_share_router_without_swapping(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    engine._duplex_control_client = None
    fence = DuplexFence("sid")
    duplex = AppendDuplexInputMessage(
        control_id="duplex-control",
        fence=fence,
        session_id="sid",
        mode="audio",
        payload=b"audio",
    )
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="collective-rpc"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        duplex_result = executor.submit(engine._get_duplex_control_client().execute, duplex, timeout=1)
        collective_result = executor.submit(engine.collective_rpc, "health", timeout=1)
        submitted_types = {request_q.get(timeout=1).type, request_q.get(timeout=1).type}
        assert submitted_types == {"append_duplex_input", "collective_rpc"}
        rpc_q.put(
            CollectiveRPCResultMessage(
                rpc_id="collective-rpc",
                method="health",
                stage_ids=[0],
                results=["healthy"],
            )
        )
        rpc_q.put(
            DuplexControlResultMessage(
                control_id="duplex-control",
                fence=fence,
                operation="append",
                session_id="sid",
                ok=True,
                stage_results=[],
            )
        )

        assert collective_result.result(timeout=1) == ["healthy"]
        assert duplex_result.result(timeout=1)["fence"] == fence

    engine._correlated_rpc_client.close()


def test_collective_rpc_preserves_request_queue_backpressure(mocker: MockerFixture):
    class SignallingQueue(queue.Queue):
        def __init__(self) -> None:
            super().__init__(maxsize=1)
            self.put_attempted = threading.Event()

        def put(self, item, block=True, timeout=None):
            self.put_attempted.set()
            return super().put(item, block=block, timeout=timeout)

    request_q = SignallingQueue()
    request_q.put("queue-is-full")
    request_q.put_attempted.clear()
    rpc_q = queue.Queue()
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    engine._duplex_control_client = None
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="blocked-rpc"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(engine.collective_rpc, "health", timeout=1)
        assert request_q.put_attempted.wait(timeout=1)
        assert not pending.done()

        assert request_q.get(timeout=1) == "queue-is-full"
        assert request_q.get(timeout=1).rpc_id == "blocked-rpc"
        rpc_q.put(
            CollectiveRPCResultMessage(
                rpc_id="blocked-rpc",
                method="health",
                stage_ids=[0],
                results=["healthy"],
            )
        )
        assert pending.result(timeout=1) == ["healthy"]

    engine._correlated_rpc_client.close()


def test_open_duplex_session_raises_on_stage_control_error(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    control_result = DuplexControlResultMessage(
        control_id="ctrl-error",
        fence=DuplexFence("sid"),
        operation="open",
        session_id="sid",
        ok=False,
        stage_results=[{"stage_id": 0, "replica_id": 0, "result": {"supported": False, "error": "boom"}}],
        unsupported_count=1,
        error_count=1,
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    engine._duplex_control_client = None
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="ctrl-error"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(engine.open_duplex_session, "sid", fence=DuplexFence("sid"), timeout=1)
        request_q.get(timeout=1)
        rpc_q.put(control_result)
        with pytest.raises(RuntimeError, match="duplex open failed"):
            pending.result(timeout=1)
    engine._correlated_rpc_client.close()


def test_shutdown_does_not_race_runtime_cleanup_with_live_orchestrator(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine._weak_finalizer = None
    engine.request_queue = mocker.MagicMock()
    engine.request_queue.sync_q.put.side_effect = queue.Full
    engine.output_queue = mocker.MagicMock()
    engine.rpc_output_queue = mocker.MagicMock()
    engine._correlated_rpc_client = mocker.MagicMock()
    engine.orchestrator_thread = mocker.MagicMock()
    engine.orchestrator_thread.is_alive.return_value = True
    engine._runtime = mocker.MagicMock()

    engine.shutdown()

    engine.request_queue.sync_q.put.assert_called_once()
    assert engine.request_queue.sync_q.put.call_args.kwargs["timeout"] > 0
    assert engine.orchestrator_thread.join.call_args_list[0].kwargs["timeout"] > 0
    engine._correlated_rpc_client.close.assert_called_once_with()
    engine.request_queue.close.assert_called_once_with()
    engine.output_queue.close.assert_called_once_with()
    engine.rpc_output_queue.close.assert_called_once_with()
    engine._runtime.shutdown.assert_not_called()


def test_shutdown_releases_runtime_after_orchestrator_stops(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine._weak_finalizer = None
    engine.request_queue = mocker.MagicMock()
    engine.output_queue = mocker.MagicMock()
    engine.rpc_output_queue = mocker.MagicMock()
    engine._correlated_rpc_client = mocker.MagicMock()
    engine.orchestrator_thread = mocker.MagicMock()
    engine.orchestrator_thread.is_alive.side_effect = [True, False]
    engine._runtime = mocker.MagicMock()

    engine.shutdown()

    engine.orchestrator_thread.join.assert_called_once()
    engine._runtime.shutdown.assert_called_once_with()


def test_shutdown_defers_runtime_release_until_live_orchestrator_stops(mocker: MockerFixture):
    orchestrator_stopped = threading.Event()
    runtime_released = threading.Event()

    class ControllableOrchestratorThread:
        def __init__(self) -> None:
            self.join_timeouts: list[float | None] = []

        def is_alive(self) -> bool:
            return not orchestrator_stopped.is_set()

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)
            if timeout is None:
                orchestrator_stopped.wait()

    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine._weak_finalizer = None
    engine.request_queue = mocker.MagicMock()
    engine.output_queue = mocker.MagicMock()
    engine.rpc_output_queue = mocker.MagicMock()
    engine._correlated_rpc_client = mocker.MagicMock()
    engine.orchestrator_thread = ControllableOrchestratorThread()
    engine._runtime = mocker.MagicMock()
    engine._runtime.shutdown.side_effect = runtime_released.set

    try:
        engine.shutdown()

        engine._runtime.shutdown.assert_not_called()
        orchestrator_stopped.set()
        assert runtime_released.wait(timeout=1)
        engine._runtime.shutdown.assert_called_once_with()
        assert engine.orchestrator_thread.join_timeouts[0] is not None
        assert engine.orchestrator_thread.join_timeouts[-1] is None
    finally:
        orchestrator_stopped.set()


def test_weak_shutdown_is_bounded_when_request_queue_is_full(mocker: MockerFixture):
    request_queue = mocker.MagicMock()
    request_queue.sync_q.put.side_effect = queue.Full
    output_queue = mocker.MagicMock()
    rpc_output_queue = mocker.MagicMock()
    router = mocker.MagicMock()
    orchestrator_thread = mocker.MagicMock()
    orchestrator_thread.is_alive.return_value = True

    weak_shutdown_async_omni_engine(
        orchestrator_thread,
        request_queue,
        output_queue,
        rpc_output_queue,
        router,
    )

    request_queue.sync_q.put.assert_called_once()
    assert request_queue.sync_q.put.call_args.kwargs["timeout"] > 0
    orchestrator_thread.join.assert_called_once()
    assert orchestrator_thread.join.call_args.kwargs["timeout"] > 0
    request_queue.close.assert_called_once_with()
    output_queue.close.assert_called_once_with()
    rpc_output_queue.close.assert_called_once_with()
