# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Event-driven orchestration loop tests (the default loop).

The scenario matrix in ``test_orchestrator.py`` and
``test_orchestrator_error_handling.py`` runs on both loops through their
parametrized ``orchestrator_factory``. This module holds the coverage specific
to the event-driven loop: reader reconcile on client swap and replica attach,
final-output delivery, and flag parsing.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import janus
import pytest

from vllm_omni.engine import orchestrator as orchestrator_module
from vllm_omni.engine.async_engine_utils import OutputQueueSink
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ShutdownRequestMessage
from vllm_omni.engine.orchestrator import (
    _event_driven_orch_enabled,
)

from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    OrchestratorFixture,
    RecordingOutputProcessor,
    _build_harness,
    _build_request_output,
    _engine_core_outputs,
    _enqueue_add_request,
    _get_output_message,
    _sampling_params,
    _shutdown_orchestrator,
    _wait_for,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def orchestrator_factory(monkeypatch):
    """Harness factory pinned to the event-driven loop."""
    monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", "1")
    fixtures: list[OrchestratorFixture] = []

    def _factory(*args, **kwargs) -> OrchestratorFixture:
        fixture = _build_harness(*args, **kwargs)
        assert fixture.orchestrator._event_driven_orch is True
        fixtures.append(fixture)
        return fixture

    yield _factory

    for fixture in fixtures:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


# ---------------------------------------------------------------------------
# Event-driven-specific behavior
# ---------------------------------------------------------------------------


def test_flag_parsing(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", raising=False)
    assert _event_driven_orch_enabled() is True
    for value in ("1", "true", "True", "YES", "on"):
        monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", value)
        assert _event_driven_orch_enabled() is True
    for value in ("0", "false", "off", ""):
        monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", value)
        assert _event_driven_orch_enabled() is False


@pytest.mark.parametrize("value, expected", [(None, True), ("0", False)], ids=["unset", "opt-out"])
def test_default_is_event_driven_loop(monkeypatch, value, expected) -> None:
    """Without the env flag the harness runs the event-driven loop; ``0`` selects the legacy poll loop."""
    if value is None:
        monkeypatch.delenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", value)
    fixture = _build_harness([FakeStageClient(stage_type="llm", final_output=True)])
    try:
        assert fixture.orchestrator._event_driven_orch is expected
    finally:
        fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


@pytest.mark.asyncio
async def test_reader_reconcile_picks_up_swapped_client(orchestrator_factory) -> None:
    """Outputs from a replica whose client object was replaced still flow.

    The event-driven loop binds one reader task per client object; the
    periodic reconcile must respawn the reader when ``pool.clients[replica]``
    is swapped (replica replacement), otherwise the new client's outputs
    would never be drained.
    """
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    processor = FakeOutputProcessor(request_outputs=[_build_request_output("req-swap", token_ids=[3], finished=True)])
    orchestrator_fixture = orchestrator_factory([stage0], output_processors=[processor])
    request = SimpleNamespace(request_id="req-swap", prompt_token_ids=[1, 2])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-swap",
            prompt=request,
            original_prompt={"prompt": "swap"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Swap in a fresh client for the same replica slot; keep the pool's
        # other wiring intact. The reconcile tick (0.5 s) must respawn the
        # reader bound to the new client object.
        pool = orchestrator_fixture.orchestrator.stage_pools[0]
        replacement = FakeStageClient(stage_type="llm", final_output=True)
        replacement.stage_id = stage0.stage_id
        replacement.replica_id = stage0.replica_id
        pool.clients[0] = replacement

        replacement.push_engine_core_outputs(_engine_core_outputs("swapped-raw", 1.0))

        output_msg = await _get_output_message(orchestrator_fixture, timeout=5.0)
        assert output_msg.request_id == "req-swap"
        assert output_msg.finished is True
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_replica_attached_at_runtime_is_read_without_waiting_for_reconcile_tick(
    orchestrator_factory, monkeypatch
) -> None:
    """Attaching a replica wakes the dispatcher; its outputs must not wait for the periodic reconcile."""
    monkeypatch.setattr(orchestrator_module, "_ORCH_READER_RECONCILE_INTERVAL_S", 3600.0)
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    processor = RecordingOutputProcessor()
    orchestrator_fixture = orchestrator_factory([stage0], output_processors=[processor])

    try:
        pool = orchestrator_fixture.orchestrator.stage_pools[0]
        await _wait_for(lambda: pool.membership_listener is not None)
        attached = FakeStageClient(stage_type="llm", final_output=True)
        attached.replica_id = pool.add_client("tcp://replica-1", attached)
        assert attached.replica_id == 1

        attached.push_engine_core_outputs(_engine_core_outputs("attached-raw", 1.0))

        await _wait_for(lambda: len(processor.process_calls) == 1)
        assert [output.request_id for output in processor.process_calls[0][0][0]] == ["attached-raw"]
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)
    assert pool.membership_listener is None


# ---------------------------------------------------------------------------
# Final-output delivery (OutputQueueSink -> AsyncOmniEngine.get_outputs_async)
# ---------------------------------------------------------------------------


def _drain_engine(alive: bool = True) -> AsyncOmniEngine:
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = janus.Queue()
    engine._output_sink = OutputQueueSink(engine.output_queue.sync_q)
    engine.orchestrator_thread = SimpleNamespace(is_alive=lambda: alive)
    return engine


@pytest.mark.asyncio
async def test_output_drain_caps_the_batch() -> None:
    engine = _drain_engine()
    try:
        assert await engine.get_outputs_async(timeout=0.01) == []
        for i in range(5):
            engine._output_sink.put_nowait(i)
        assert await engine.get_outputs_async(timeout=1.0, max_messages=3) == [0, 1, 2]
        assert await engine.get_outputs_async(timeout=1.0, max_messages=3) == [3, 4]
    finally:
        engine.output_queue.close()


@pytest.mark.asyncio
async def test_output_drain_wakes_on_put_from_another_thread() -> None:
    """The orchestrator thread's put wakes the waiting serving loop, no polling."""
    engine = _drain_engine()
    try:
        timer = threading.Timer(0.05, engine._output_sink.put_nowait, args=("late-msg",))
        timer.start()
        loop = asyncio.get_running_loop()
        started = loop.time()
        msgs = await engine.get_outputs_async(timeout=5.0)
        timer.join()
        assert msgs == ["late-msg"]
        assert loop.time() - started < 1.0
    finally:
        engine.output_queue.close()


@pytest.mark.asyncio
async def test_output_drain_cancel_leaves_queued_messages() -> None:
    engine = _drain_engine(alive=True)
    try:
        reader = asyncio.ensure_future(engine.get_outputs_async(timeout=5.0))
        await asyncio.sleep(0.02)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        engine._output_sink.put_nowait("kept")
        assert await engine.get_outputs_async(timeout=1.0) == ["kept"]
    finally:
        engine.output_queue.close()


@pytest.mark.asyncio
async def test_output_drain_delivers_unannounced_message_before_dead_check() -> None:
    """A message queued without a wakeup is returned at timeout, not lost to the dead check."""
    engine = _drain_engine(alive=False)
    try:
        reader = asyncio.ensure_future(engine.get_outputs_async(timeout=0.1))
        await asyncio.sleep(0.02)  # reader is now waiting for a wakeup
        engine.output_queue.sync_q.put_nowait("fatal-error")  # bypasses the sink
        assert await reader == ["fatal-error"]
    finally:
        engine.output_queue.close()


@pytest.mark.asyncio
async def test_output_drain_raises_when_orchestrator_dead() -> None:
    engine = _drain_engine(alive=False)
    try:
        with pytest.raises(RuntimeError, match="Orchestrator died"):
            await engine.get_outputs_async(timeout=0.05)
    finally:
        engine.output_queue.close()
