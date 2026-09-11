# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``DuplexOrchestrator``: the template seams and the stage port around ``OrchestratorBase``."""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.engine.test_orchestrator import (
    FakeOutputProcessor,
    FakeRunningCounter,
    FakeStageClient,
    _build_stage_pools,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator, DuplexOrchestratorRequestState
from vllm_omni.engine.messages import AbortRequestMessage, ShutdownRequestMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "duplex-orch"


def _encode_audio(audio: object, sample_rate_hz: int, response_format: str, speed: float | None) -> str | None:
    del sample_rate_hz, response_format, speed
    samples = int(np.asarray(audio, dtype=np.float32).size) if audio is not None else 0
    return f"wav-{samples}" if samples > 0 else None


def _stage_configs(count: int) -> list[object]:
    return [SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)) for _ in range(count)]


def _build(
    *,
    stages: int = 1,
    running_counter: FakeRunningCounter | None = None,
    runtime_config: DuplexSessionRuntimeConfig | None = None,
) -> tuple[DuplexOrchestrator, list[FakeStageClient], asyncio.Queue, asyncio.Queue]:
    clients = [FakeStageClient(stage_type="llm", final_output=index == stages - 1) for index in range(stages)]
    pools = _build_stage_pools(
        [[client] for client in clients],
        output_processors=[FakeOutputProcessor() for _ in clients],
        stage_vllm_configs=_stage_configs(stages),
    )
    rpc_q: asyncio.Queue = asyncio.Queue()
    output_q: asyncio.Queue = asyncio.Queue()
    orchestrator = DuplexOrchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=output_q,
        rpc_async_queue=rpc_q,
        stage_pools=pools,
        running_counter=running_counter,
        plugin=MiniCPMO45DuplexPlugin(_encode_audio),
        duplex_session_config=runtime_config or DuplexSessionRuntimeConfig(reaper_interval_s=0.01),
        model_config=None,
    )
    return orchestrator, clients, rpc_q, output_q


def _open_message(extra_body: dict[str, object] | None = None) -> OpenDuplexSessionMessage:
    return OpenDuplexSessionMessage(
        control_id=f"open-{SESSION_ID}",
        session_id=SESSION_ID,
        session_config=DuplexSessionConfig(
            model="openbmb/MiniCPM-o-4_5",
            modalities=["text"],
            instructions="public instructions",
            extra_body={"auto_response": True, **(extra_body or {})},
        ),
    )


async def _open(orchestrator: DuplexOrchestrator, rpc_q: asyncio.Queue, **kwargs: Any) -> DuplexControlResultMessage:
    await orchestrator._dispatch_message(_open_message(**kwargs))
    result = await asyncio.wait_for(rpc_q.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage)
    return result


async def _close(orchestrator: DuplexOrchestrator, rpc_q: asyncio.Queue) -> DuplexControlResultMessage:
    await orchestrator._dispatch_message(
        CloseDuplexSessionMessage(control_id=f"close-{SESSION_ID}", session_id=SESSION_ID, reason="test")
    )
    result = await asyncio.wait_for(rpc_q.get(), timeout=2.0)
    await _settle(orchestrator)
    return result


def _stage0_request_id(epoch: int = 0) -> str:
    return duplex_resource_request_id(DuplexFence(SESSION_ID, epoch=epoch), "stage0")


def _append_audio(samples: int = 16000) -> commands.AppendAudio:
    return commands.AppendAudio(
        audio=struct.pack(f"<{samples}f", *([0.05] * samples)),
        format="pcm_f32le",
        sample_rate_hz=16000,
        is_speech=True,
    )


async def _submit(orchestrator: DuplexOrchestrator, command: commands.DuplexCommand) -> None:
    assert await orchestrator._dispatch_message(DuplexSessionCommandMessage(session_id=SESSION_ID, command=command))
    await _settle(orchestrator)


async def _settle(orchestrator: DuplexOrchestrator, *, timeout_s: float = 3.0) -> None:
    """Let the session runner drain its mailbox and append tasks."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    quiet = 0
    while loop.time() < deadline:
        runner = orchestrator.session_manager.runners.get(SESSION_ID)
        busy = runner is not None and (
            not runner._mailbox.empty()
            or any(not task.done() for task in runner.tasks.append_tasks)
            or any(not task.done() for task in runner._background_tasks)
        )
        busy = busy or any(not task.done() for task in orchestrator.session_manager._dispatched_control_tasks)
        quiet = 0 if busy else quiet + 1
        if quiet >= 5:
            return
        await asyncio.sleep(0.01)


def _tts_output(request_id: str, *, samples: int = 24000, text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text=text, token_ids=[], multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(samples, dtype=np.float32),
            "sr": 24000,
            "meta.duplex_turn_id": np.array([0], dtype=np.int32),
            "meta.duplex_epoch": np.array([0], dtype=np.int32),
        },
    )


# --------------------------------------------------------------------------- #
# Message dispatch and session lifecycle                                      #
# --------------------------------------------------------------------------- #


def test_turn_based_orchestrator_has_no_session_manager() -> None:
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=[],
    )
    assert not hasattr(orchestrator, "session_manager")


@pytest.mark.asyncio
async def test_duplex_messages_are_dispatched_and_generic_ones_are_not() -> None:
    orchestrator, _, rpc_q, _ = _build()
    assert await orchestrator._dispatch_message(AbortRequestMessage(request_ids=["x"])) is False
    assert await orchestrator._dispatch_message(ShutdownRequestMessage()) is False
    assert await orchestrator._dispatch_message(_open_message()) is True
    assert (await asyncio.wait_for(rpc_q.get(), timeout=2.0)).ok is True
    await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_open_preregisters_the_stage0_request_and_close_releases_it() -> None:
    counter = FakeRunningCounter()
    orchestrator, clients, rpc_q, _ = _build(running_counter=counter)
    result = await _open(orchestrator, rpc_q)
    assert result.ok is True
    assert result.public_session["id"] == SESSION_ID
    assert result.capabilities is not None and result.capabilities.supports_input_append

    request_id = _stage0_request_id()
    request_state = orchestrator.request_states[request_id]
    assert isinstance(request_state, DuplexOrchestratorRequestState)
    assert request_state.session_owned is True
    assert request_state.session_id == SESSION_ID
    assert request_state.fence == DuplexFence(SESSION_ID)
    assert request_state.streaming.enabled is True
    # Preregistration reserves the id; nothing is running until an append submits.
    assert counter.value == 0
    assert clients[0].add_request_calls == []
    assert orchestrator.session_manager.runner_for_request_id(request_id) is not None

    closed = await _close(orchestrator, rpc_q)
    assert closed.ok is True
    assert request_id not in orchestrator.request_states
    assert orchestrator.session_manager.runner_for_request_id(request_id) is None
    assert counter.value == 0
    assert orchestrator.session_manager.active_count() == 0


@pytest.mark.asyncio
async def test_open_failure_rolls_back_session_and_reserved_request() -> None:
    orchestrator, _, rpc_q, _ = _build()
    # Server-owned runtime keys may not come from the client.
    result = await _open(orchestrator, rpc_q, extra_body={"duplex_stage_max_tokens": {"0": 99}})

    assert result.ok is False
    assert result.error_code == "invalid_duplex_runtime_config"
    assert orchestrator.request_states == {}
    assert orchestrator.session_manager.active_count() == 0


@pytest.mark.asyncio
async def test_open_fails_cleanly_when_a_stage_has_no_live_replica() -> None:
    orchestrator, _, rpc_q, _ = _build(stages=2)
    orchestrator.stage_pools[1].evict_replica(0)

    result = await _open(orchestrator, rpc_q)

    assert result.ok is False
    assert "stage 1 has no live replica" in (result.error_message or "")
    assert orchestrator.request_states == {}
    assert orchestrator.session_manager.active_count() == 0


@pytest.mark.asyncio
async def test_bridge_state_keeps_public_and_runtime_config_separate() -> None:
    orchestrator, _, rpc_q, _ = _build()
    await _open(orchestrator, rpc_q)
    request_state = orchestrator.request_states[_stage0_request_id()]
    bridge = request_state.streaming.bridge_states["duplex"]

    assert bridge["session_id"] == SESSION_ID
    assert bridge["fence"] == DuplexFence(SESSION_ID)
    assert (bridge["epoch"], bridge["turn_id"], bridge["model_turn_id"]) == (0, 0, 0)
    assert "incarnation" not in bridge
    assert bridge["session_config"]["instructions"] == "public instructions"
    assert "duplex_stage_sampling_params" not in bridge["session_config"]
    assert bridge["runtime_config"]["instructions"] == "public instructions"
    assert bridge["runtime_config"]["duplex_stage_max_tokens"] == {"0": 20, "1": 8192}
    # The plugin's sampling policy is applied to the preregistered request.
    assert request_state.sampling_params_list[0].max_tokens == 20
    await orchestrator.session_manager.shutdown()


# --------------------------------------------------------------------------- #
# Stage port                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_append_submits_the_resumable_stage0_request_and_counts_it_running() -> None:
    counter = FakeRunningCounter()
    orchestrator, clients, rpc_q, _ = _build(running_counter=counter)
    await _open(orchestrator, rpc_q)
    request_id = _stage0_request_id()

    await _submit(orchestrator, _append_audio())
    assert len(clients[0].add_request_calls) == 1
    submitted = clients[0].add_request_calls[0][0]
    assert submitted.request_id == request_id
    assert submitted.resumable is True
    assert submitted.sampling_params.max_tokens == 20
    request_state = orchestrator.request_states[request_id]
    assert request_state.stage_fences[0] == DuplexFence(SESSION_ID)
    assert 0 in request_state.stage_submit_ts
    assert counter.value == 1
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None and session.stage_request_submitted(0, request_id)

    # Later units update the same resumable request; the counter stays at one.
    await _submit(orchestrator, _append_audio())
    assert len(clients[0].add_request_calls) == 2
    assert clients[0].add_request_calls[1][0].request_id == request_id
    assert counter.value == 1

    await _close(orchestrator, rpc_q)
    assert clients[0].abort_calls == [[request_id]]
    assert counter.value == 0


@pytest.mark.asyncio
async def test_session_update_refreshes_the_next_append_sampling_params() -> None:
    orchestrator, clients, rpc_q, _ = _build()
    await _open(orchestrator, rpc_q)
    request_state = orchestrator.request_states[_stage0_request_id()]
    assert request_state.sampling_params_list[0].max_tokens == 20

    await _submit(orchestrator, commands.UpdateSession(patch={"max_output_tokens": 7}))
    await _submit(orchestrator, _append_audio())

    assert request_state.sampling_params_list[0].max_tokens == 7
    assert clients[0].add_request_calls[-1][0].sampling_params.max_tokens == 7
    await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_forwarded_stage_requests_are_bound_and_barge_in_aborts_them() -> None:
    orchestrator, clients, rpc_q, _ = _build(stages=2)
    await _open(orchestrator, rpc_q)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    request_state = orchestrator.request_states[request_id]
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None

    # The base forwards Stage0 text to the TTS stage and reports the submission.
    forwarded = SimpleNamespace(request_id=request_id, prompt_token_ids=[1, 2], resumable=True)
    replica_id = await orchestrator.stage_pools[1].submit_initial(request_id, request_state, forwarded)
    orchestrator._on_stage_submitted(1, request_id, replica_id, request_state)
    assert request_state.stage_fences[1] == DuplexFence(SESSION_ID)
    assert session.stage_request_submitted(1, request_id)

    await _submit(orchestrator, commands.BargeIn())
    assert session.epoch == 1
    assert clients[0].abort_calls == [[request_id]]
    assert clients[1].abort_calls == [[request_id]]
    assert request_id not in orchestrator.request_states
    assert session.resource_request_ids() == []
    await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_session_owned_outputs_reach_the_runner_and_never_the_client_queue() -> None:
    orchestrator, _, rpc_q, output_q = _build(stages=2)
    await _open(orchestrator, rpc_q)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    request_state = orchestrator.request_states[request_id]
    while not output_q.empty():
        output_q.get_nowait()

    consumed = await orchestrator._intercept_stage_output(1, 0, _tts_output(request_id), request_state, None, None)
    await _settle(orchestrator)

    assert consumed is True
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None and session.active_response_id is not None
    types = [message.event.type for message in [output_q.get_nowait() for _ in range(output_q.qsize())]]
    assert "response.created" in types and "response.output_audio.delta" in types

    orphan = DuplexOrchestratorRequestState(
        request_id="duplex-s.b3RoZXI.e.0.r.stage0",
        prompt=None,
        sampling_params_list=[],
        final_stage_id=1,
        session_owned=True,
        session_id="other",
        fence=DuplexFence("other"),
    )
    assert await orchestrator._intercept_stage_output(1, 0, _tts_output(orphan.request_id), orphan, None, None)
    await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_forward_failure_closes_the_owning_session() -> None:
    orchestrator, clients, rpc_q, output_q = _build(stages=2)
    await _open(orchestrator, rpc_q)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    request_state = orchestrator.request_states[request_id]
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None

    absorbed = await orchestrator._handle_forward_failure(request_id, 1, request_state, ValueError("bad thinker"))
    await _settle(orchestrator)

    assert absorbed is True
    assert request_id not in orchestrator.request_states
    assert clients[0].abort_calls == [[request_id]]
    assert SESSION_ID not in orchestrator.session_manager.runners
    assert session.state == DuplexSessionState.CLOSED
    assert orchestrator.session_manager.active_count() == 0
    types = [message.event.type for message in [output_q.get_nowait() for _ in range(output_q.qsize())]]
    # The stage failure is reported before the session expires, never after.
    assert types.index("error") < types.index("session.expired")
    assert types[-1] == "session.expired"


@pytest.mark.asyncio
async def test_request_cleanup_failure_is_deferred_and_retried_by_the_reaper(monkeypatch) -> None:
    orchestrator, clients, rpc_q, _ = _build()
    await _open(orchestrator, rpc_q)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    manager = orchestrator.session_manager

    async def failing_abort(request_ids: list[str]) -> None:
        raise RuntimeError("stage abort failed")

    monkeypatch.setattr(clients[0], "abort_requests_async", failing_abort)
    with pytest.raises(RuntimeError, match="stage abort failed"):
        await orchestrator._cleanup_request_ids([request_id], abort=True, release_owners=True)
    await _settle(orchestrator)

    assert SESSION_ID not in manager.runners
    assert SESSION_ID in manager._closing  # admission slot retained
    assert manager._pending_request_cleanups and not manager._request_cleanups_in_progress
    assert manager.active_count() == 0 and manager._admission_count() == 1

    monkeypatch.undo()
    assert await manager.reap_expired() >= 1
    assert manager._pending_request_cleanups == {}
    assert manager._admission_count() == 0
    assert request_id not in orchestrator.request_states


@pytest.mark.asyncio
async def test_reaper_runs_as_a_background_task_and_survives_one_failure(monkeypatch) -> None:
    orchestrator, _, _, _ = _build(runtime_config=DuplexSessionRuntimeConfig(reaper_interval_s=0.01))
    calls = {"count": 0}

    async def reap_expired(now: float | None = None) -> int:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient cleanup failure")
        return 0

    monkeypatch.setattr(orchestrator.session_manager, "reap_expired", reap_expired)
    tasks = orchestrator._background_tasks()
    assert len(tasks) == 1
    task = asyncio.create_task(tasks[0])
    await asyncio.sleep(0.05)
    orchestrator._shutdown_event.set()
    await task
    assert calls["count"] >= 2
    await orchestrator._shutdown_extensions()
