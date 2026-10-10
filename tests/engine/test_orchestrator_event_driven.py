# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Event-driven orchestration loop (``VLLM_OMNI_EVENT_DRIVEN_ORCH=1``) tests.

Parity suite: re-runs the legacy orchestration scenarios from
``test_orchestrator.py`` / ``test_orchestrator_error_handling.py`` with the
event-driven loop selected, so both loops are held to the same behavior. Plus
event-driven-specific coverage: reader reconcile on client swap, the blocking
final-output drain, and flag parsing.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import janus
import pytest

from vllm_omni.engine.async_engine_utils import OutputQueueSink
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ShutdownRequestMessage
from vllm_omni.engine.orchestrator import (
    _event_driven_orch_default_for_pipeline,
    _event_driven_orch_enabled,
)

from . import test_orchestrator as legacy
from . import test_orchestrator_error_handling as legacy_errors
from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    OrchestratorFixture,
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
    """Flag-setting clone of the legacy harness fixture.

    Sets ``VLLM_OMNI_EVENT_DRIVEN_ORCH=1`` before any Orchestrator is
    constructed and asserts the flag actually took effect, so the parity tests
    cannot silently exercise the legacy poll loop.
    """
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
# Parity: the legacy scenario matrix, re-run through the event-driven loop
# ---------------------------------------------------------------------------

_PARITY_TESTS = [
    legacy.test_run_two_stage_llm,
    legacy.test_run_single_stage_diffusion,
    legacy.test_run_single_stage_diffusion_streaming_forwards_intermediate_chunks,
    legacy.test_run_llm_to_diffusion,
    legacy.test_run_async_chunk,
    legacy.test_run_shutdown,
    legacy.test_run_abort,
    legacy.test_multi_replica_round_robin_distribution,
    legacy.test_multi_replica_abort_broadcasts_to_all_replicas,
    legacy.test_multi_replica_shutdown_all_replicas,
    legacy.test_multi_replica_cfg_companion_inherits_parent_affinity,
    # Stats plumbing. The scheduler-stats cases matter most here: a batch with
    # no request outputs still carries SchedulerStats on throttled ticks, and a
    # reader that drops every output-less batch silently stops reporting
    # KV/queue gauges under the event-driven loop.
    legacy.test_orchestrator_records_iteration_stats_without_scheduler_stats,
    legacy.test_orchestrator_records_scheduler_stats_without_outputs,
    legacy.test_orchestrator_does_not_build_iteration_stats_for_finished_only_batch,
    legacy.test_orchestrator_does_not_build_iteration_stats_without_stat_logger,
    # Per-replica fault isolation (#4285): a dead replica must be evicted and
    # the server kept up, on both the LLM reader path and the diffusion poller.
    legacy_errors.test_engine_dead_error_evicts_replica_and_keeps_running,
    legacy_errors.test_engine_dead_error_fails_only_dead_replica_requests,
    legacy_errors.test_forward_to_dead_downstream_stage_fails_request_not_server,
    legacy_errors.test_add_request_to_dead_stage_fails_request_not_server,
    legacy_errors.test_diffusion_replica_death_on_poll_keeps_server,
    legacy_errors.test_diffusion_error_output_routed_as_finished,
    legacy_errors.test_diffusion_client_error_output_propagates_status_code,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_test", _PARITY_TESTS, ids=lambda f: f.__name__)
async def test_event_driven_parity(legacy_test, orchestrator_factory) -> None:
    await legacy_test(orchestrator_factory)


# ---------------------------------------------------------------------------
# Event-driven-specific behavior
# ---------------------------------------------------------------------------


def test_flag_parsing(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", raising=False)
    assert _event_driven_orch_enabled() is False
    for value in ("1", "true", "True", "YES", "on"):
        monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", value)
        assert _event_driven_orch_enabled() is True
    for value in ("0", "false", "off", ""):
        monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", value)
        assert _event_driven_orch_enabled() is False


def test_only_qwen3_tts_has_a_pipeline_default() -> None:
    assert _event_driven_orch_default_for_pipeline("qwen3_tts") is True
    for model_type in (None, "qwen3_omni_moe", "minicpmo_4_5", "moss_tts_delay"):
        assert _event_driven_orch_default_for_pipeline(model_type) is False


def test_explicit_event_driven_value_overrides_pipeline_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", raising=False)
    assert _event_driven_orch_enabled(default=True) is True
    monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", "0")
    assert _event_driven_orch_enabled(default=True) is False


def test_default_is_legacy_loop(monkeypatch) -> None:
    """Without the env flag, the harness runs the legacy poll loop."""
    monkeypatch.delenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", raising=False)
    fixture = _build_harness([FakeStageClient(stage_type="llm", final_output=True)])
    try:
        assert fixture.orchestrator._event_driven_orch is False
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
async def test_output_drain_coalesces_wakeups_while_busy(monkeypatch) -> None:
    """Puts made after a wakeup is scheduled schedule no further loop callbacks."""
    engine = _drain_engine()
    sink = engine._output_sink
    loop = asyncio.get_running_loop()
    try:
        assert await engine.get_outputs_async(timeout=0.01) == []  # armed, nothing queued
        calls = []
        original = loop.call_soon_threadsafe

        def counting_call_soon_threadsafe(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(loop, "call_soon_threadsafe", counting_call_soon_threadsafe)
        for i in range(10):
            sink.put_nowait(i)
        assert len(calls) == 1
        assert await engine.get_outputs_async(timeout=1.0) == list(range(10))
    finally:
        engine.output_queue.close()


@pytest.mark.asyncio
async def test_output_drain_keeps_messages_put_after_a_timeout() -> None:
    engine = _drain_engine(alive=True)
    try:
        assert await engine.get_outputs_async(timeout=0.02) == []
        engine._output_sink.put_nowait("after-timeout")
        assert await engine.get_outputs_async(timeout=1.0) == ["after-timeout"]
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
