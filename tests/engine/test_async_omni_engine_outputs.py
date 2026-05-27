"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies,
subsequent attempts to collect output raise RuntimeError.
"""

import queue
from types import SimpleNamespace

import janus
import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
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
    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get_nowait.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, mocker, thread_alive=True)

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

    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get_nowait.return_value = fatal_msg

    engine = _make_engine(mock_queue, mocker, thread_alive=False)

    msg = await engine.try_get_output_async()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True


def test_process_single_result_preserves_nonfinal_stage_output(mocker: MockerFixture):
    """Public OmniRequestOutput must keep incremental final-stage chunks open."""
    omni = object.__new__(AsyncOmni)
    omni._enable_ar_profiler = False
    omni.engine = mocker.MagicMock()
    omni.engine.get_stage_metadata.return_value = SimpleNamespace(
        final_output=True,
        final_output_type="audio",
    )

    metrics = SimpleNamespace(
        stage_events={},
        stage_first_ts=[None],
        stage_last_ts=[None],
        e2e_done=set(),
        on_finalize_request=mocker.MagicMock(),
        on_stage_metrics=mocker.MagicMock(),
    )
    stage_output = SimpleNamespace(
        request_id="r1",
        finished=False,
        final_output_type="audio",
        stage_durations={},
        peak_memory_mb=0.0,
    )
    result = OutputMessage(
        request_id="r1",
        stage_id=0,
        engine_outputs=stage_output,
        metrics=None,
        finished=False,
    )

    output = omni._process_single_result(
        result,
        stage_id=0,
        metrics=metrics,
        req_start_ts={"r1": 1.0},
        wall_start_ts=1.0,
        final_stage_id_for_e2e=0,
    )

    assert output is not None
    assert output.finished is False
    metrics.on_finalize_request.assert_not_called()


@pytest.mark.asyncio
async def test_async_omni_abort_ignores_closed_engine_queue(mocker: MockerFixture):
    """Closing an async generator during shutdown should not leak janus errors."""
    omni = object.__new__(AsyncOmni)
    omni.engine = mocker.MagicMock()
    omni.engine.abort_async = mocker.AsyncMock(side_effect=janus.SyncQueueShutDown)
    omni.request_states = {"r1": object()}
    omni.log_stats = False

    await omni.abort("r1")

    assert "r1" not in omni.request_states
