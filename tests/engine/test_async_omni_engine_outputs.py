"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies it enqueues
a fatal ``ErrorMessage`` and then shuts the output queue down. Readers must
drain the fatal message first (so the caller sees ``fatal=True``) and then
raise ``RuntimeError`` instead of hanging — including a reader already parked on
an empty queue, which ``shutdown()`` wakes.
"""

import asyncio
import concurrent.futures
import queue

import janus
import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(output_queue, mocker: MockerFixture, *, thread_alive: bool = True) -> AsyncOmniEngine:
    """Create an AsyncOmniEngine bypassing __init__ (with a mocked queue)."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = output_queue
    engine.orchestrator_thread = mocker.MagicMock(
        is_alive=mocker.MagicMock(return_value=thread_alive),
    )
    return engine


def _make_real_engine(output_queue) -> AsyncOmniEngine:
    """Engine wired to a real janus queue so the async tests exercise the real
    ``async_q.get()`` / ``shutdown()`` path across the sync/async boundary."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = output_queue
    return engine


def _safe_shutdown(q: janus.Queue) -> None:
    """Idempotent shutdown for test cleanup (a test may have shut it down)."""
    try:
        q.shutdown()
    except Exception:
        pass


# ----------------------------- sync: try_get_output -----------------------------


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


def test_try_get_output_raises_on_queue_shutdown():
    """A real shut-down output queue must surface as RuntimeError, not leak the
    raw janus exception. Uses a real janus queue (not a mocked side_effect): the
    sync side raises ``janus.ShutDown``, which differs from the async side's
    ``janus.QueueShutDown``, so both must be mapped."""
    real_queue = janus.Queue()
    engine = _make_real_engine(real_queue)
    real_queue.shutdown()

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output(timeout=0.1)
    _safe_shutdown(real_queue)


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


# -------------------------- async: try_get_output_async --------------------------


@pytest.mark.asyncio
async def test_try_get_output_async_is_event_driven():
    """The async reader parks on an empty queue and wakes the instant a producer
    enqueues, rather than busy-polling. Uses a real janus queue across the sync
    (producer) / async (consumer) boundary, mirroring production: the
    orchestrator puts via ``sync_q`` and the engine consumes via ``async_q``.
    """
    real_queue = janus.Queue()
    engine = _make_real_engine(real_queue)
    try:
        getter = asyncio.ensure_future(engine.try_get_output_async())
        await asyncio.sleep(0.05)
        # Nothing queued yet: the coroutine must still be parked, not return.
        assert not getter.done()

        produced = OutputMessage(
            request_id="r2",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r2"),
            finished=False,
        )
        real_queue.sync_q.put_nowait(produced)

        result = await asyncio.wait_for(getter, timeout=1.0)
        assert result.request_id == "r2"

        # An already-queued message is returned promptly on the next call.
        real_queue.sync_q.put_nowait(produced)
        again = await asyncio.wait_for(engine.try_get_output_async(), timeout=1.0)
        assert again.request_id == "r2"
    finally:
        _safe_shutdown(real_queue)


@pytest.mark.asyncio
async def test_try_get_output_async_drains_fatal_then_raises_on_shutdown():
    """Mirrors a crash: the orchestrator enqueues a fatal ErrorMessage and then
    shuts the queue down. The reader must drain the fatal message first (so the
    caller can detect ``fatal=True``), then raise RuntimeError on the next read.
    """
    real_queue = janus.Queue()
    engine = _make_real_engine(real_queue)
    try:
        real_queue.sync_q.put_nowait(ErrorMessage(error="Orchestrator thread crashed", fatal=True))
        real_queue.shutdown()  # immediate=False: queued items drain before QueueShutDown

        msg = await asyncio.wait_for(engine.try_get_output_async(), timeout=1.0)
        assert msg is not None
        assert msg.type == "error"
        assert msg.fatal is True
        assert "crashed" in msg.error

        with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
            await asyncio.wait_for(engine.try_get_output_async(), timeout=1.0)
    finally:
        _safe_shutdown(real_queue)


@pytest.mark.asyncio
async def test_shutdown_wakes_parked_getter():
    """The key guarantee behind dropping the is_alive() precheck: a reader
    already parked on an EMPTY queue is woken by shutdown() — i.e. a crash that
    enqueued nothing still surfaces as RuntimeError instead of hanging forever.
    """
    real_queue = janus.Queue()
    engine = _make_real_engine(real_queue)
    try:
        getter = asyncio.ensure_future(engine.try_get_output_async())
        await asyncio.sleep(0.05)
        assert not getter.done()  # parked, nothing to read

        real_queue.shutdown()  # orchestrator died without enqueuing anything

        with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
            await asyncio.wait_for(getter, timeout=1.0)
    finally:
        _safe_shutdown(real_queue)


# --------------- bootstrap: death signaling on ANY thread exit ---------------


def _bootstrap_engine(exc: BaseException, mocker: MockerFixture) -> AsyncOmniEngine:
    """Engine whose stage init raises ``exc``, wired to real janus queues, so
    ``_bootstrap_orchestrator``'s except/finally death-signaling actually runs."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = janus.Queue()
    engine.rpc_output_queue = janus.Queue()
    engine._initialize_stages = mocker.MagicMock(side_effect=exc)
    return engine


@pytest.mark.parametrize("exc", [SystemExit("boom"), asyncio.CancelledError()])
def test_bootstrap_shuts_queues_when_thread_exits_via_base_exception(exc, mocker: MockerFixture):
    """A BaseException (SystemExit / CancelledError) skips ``except Exception``,
    so ``finally`` must still shut the queues down — otherwise a parked
    ``await async_q.get()`` would hang forever. No fatal message is enqueued."""
    engine = _bootstrap_engine(exc, mocker)
    startup_future: concurrent.futures.Future = concurrent.futures.Future()

    with pytest.raises(type(exc)):
        engine._bootstrap_orchestrator(0, startup_future)

    # Both queues shut down -> a reader unparks with RuntimeError, not a hang.
    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output(timeout=0.1)
    with pytest.raises(janus.ShutDown):  # sync side raises ShutDown, not QueueShutDown
        engine.rpc_output_queue.sync_q.get_nowait()


def test_bootstrap_enqueues_fatal_then_shuts_down_on_exception(mocker: MockerFixture):
    """A regular Exception still surfaces the real error as a fatal message
    (drained first), then shuts the queue down."""
    engine = _bootstrap_engine(RuntimeError("stage boom"), mocker)
    startup_future: concurrent.futures.Future = concurrent.futures.Future()

    with pytest.raises(RuntimeError, match="stage boom"):
        engine._bootstrap_orchestrator(0, startup_future)

    # Fatal message drains first (caller sees the real error) ...
    msg = engine.output_queue.sync_q.get_nowait()
    assert isinstance(msg, ErrorMessage) and msg.fatal
    assert "stage boom" in msg.error
    # ... then the queue is shut down (sync side raises ShutDown).
    with pytest.raises(janus.ShutDown):
        engine.output_queue.sync_q.get_nowait()
