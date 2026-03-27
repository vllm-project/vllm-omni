"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies,
subsequent attempts to collect output raise RuntimeError.
"""

import queue
from unittest.mock import MagicMock

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(output_queue, *, thread_alive: bool = True) -> AsyncOmniEngine:
    """Create an AsyncOmniEngine bypassing __init__."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = output_queue
    engine.orchestrator_thread = MagicMock(
        is_alive=MagicMock(return_value=thread_alive),
    )
    return engine


def test_try_get_output_raises_after_orchestrator_dies():
    """Draining remaining results then hitting an empty queue with a dead
    orchestrator must raise RuntimeError so callers know the pipeline is gone."""
    mock_queue = MagicMock()
    # First call succeeds; second call finds the queue empty.
    mock_queue.sync_q.get.side_effect = [
        {"type": "output", "request_id": "r1"},
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, thread_alive=True)

    # Collect the one buffered result.
    assert engine.try_get_output()["request_id"] == "r1"

    # Orchestrator thread crashes between polls.
    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output()


@pytest.mark.asyncio
async def test_try_get_output_async_raises_after_orchestrator_dies():
    """Same scenario as above but for the async variant."""
    mock_queue = MagicMock()
    mock_queue.sync_q.get_nowait.side_effect = [
        {"type": "output", "request_id": "r1"},
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, thread_alive=True)

    assert (await engine.try_get_output_async())["request_id"] == "r1"

    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        await engine.try_get_output_async()
