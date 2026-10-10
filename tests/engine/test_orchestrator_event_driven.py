# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Event-driven orchestration loop tests (the default loop).

The scenario matrix in ``test_orchestrator.py`` and
``test_orchestrator_error_handling.py`` runs on both loops through their
parametrized ``orchestrator_factory``. This module holds the coverage specific
to the event-driven loop: reader reconcile on client swap and replica attach,
the blocking final-output drain, and flag parsing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import janus
import pytest

from vllm_omni.engine import orchestrator as orchestrator_module
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
        assert processor.process_calls[0][0][0] == ["attached-raw"]
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)
    assert pool.membership_listener is None


# ---------------------------------------------------------------------------
# Blocking final-output drain (AsyncOmniEngine.get_output_blocking_async)
# ---------------------------------------------------------------------------


def _drain_engine(alive: bool = True) -> AsyncOmniEngine:
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = janus.Queue()
    engine.orchestrator_thread = SimpleNamespace(is_alive=lambda: alive)
    return engine


def _drain_cleanup(engine: AsyncOmniEngine) -> None:
    if engine._output_drain_executor is not None:
        engine._output_drain_executor.shutdown(wait=False)
        engine._output_drain_executor = None
    engine.output_queue.close()


@pytest.mark.asyncio
async def test_blocking_drain_returns_queued_message() -> None:
    engine = _drain_engine()
    try:
        engine.output_queue.sync_q.put_nowait("msg-1")
        assert await engine.get_output_blocking_async(timeout=1.0) == "msg-1"
    finally:
        _drain_cleanup(engine)


@pytest.mark.asyncio
async def test_blocking_drain_wakes_on_late_message() -> None:
    """A message put after the wait starts wakes the drain, no polling."""
    engine = _drain_engine()
    try:

        async def _delayed_put() -> None:
            await asyncio.sleep(0.05)
            engine.output_queue.sync_q.put_nowait("late-msg")

        put_task = asyncio.create_task(_delayed_put())
        msg = await engine.get_output_blocking_async(timeout=5.0)
        await put_task
        assert msg == "late-msg"
    finally:
        _drain_cleanup(engine)


@pytest.mark.asyncio
async def test_blocking_drain_timeout_returns_none_when_alive() -> None:
    engine = _drain_engine(alive=True)
    try:
        assert await engine.get_output_blocking_async(timeout=0.05) is None
    finally:
        _drain_cleanup(engine)


@pytest.mark.asyncio
async def test_blocking_drain_raises_when_orchestrator_dead() -> None:
    engine = _drain_engine(alive=False)
    try:
        with pytest.raises(RuntimeError, match="Orchestrator died"):
            await engine.get_output_blocking_async(timeout=0.05)
    finally:
        _drain_cleanup(engine)
