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
from vllm.v1.engine.exceptions import EngineDeadError

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
    plugin=None,
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
        output_sync_queue=output_q,
        rpc_async_queue=rpc_q,
        stage_pools=pools,
        running_counter=running_counter,
        plugin=plugin or MiniCPMO45DuplexPlugin(_encode_audio),
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
        output_sync_queue=asyncio.Queue(),
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
    # Set on submit from submission.resumable, not again at preregister.
    assert request_state.streaming.enabled is False
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
async def test_aura_ephemeral_preregister_disables_streaming() -> None:
    from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin

    orchestrator, _, rpc_q, _ = _build(plugin=AuraDuplexPlugin(_encode_audio))
    result = await _open(orchestrator, rpc_q)
    assert result.ok is True
    request_state = next(iter(orchestrator.request_states.values()))
    assert request_state.streaming.enabled is False
    await orchestrator.session_manager.shutdown()


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
    assert request_state.streaming.enabled is True
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
async def test_resumable_append_prompt_does_not_reach_prewarmed_stages() -> None:
    # A resumable append carries Stage0's duplex buffer. The async-chunk
    # prewarm copies ``request_state.prompt`` into downstream placeholders,
    # so that buffer must not end up there (#7962).
    orchestrator, clients, rpc_q, _ = _build(stages=2)
    orchestrator.async_chunk = True
    orchestrator._stage_receives_async_chunks = lambda stage_id: stage_id > 0  # type: ignore[method-assign]
    await _open(orchestrator, rpc_q)

    await _submit(orchestrator, _append_audio())

    assert clients[0].add_request_calls[0][0].model_intermediate_buffer
    assert len(clients[1].add_request_calls) == 1
    assert clients[1].add_request_calls[0][0].model_intermediate_buffer is None
    await orchestrator.session_manager.shutdown()


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
async def test_aura_forward_failure_keeps_the_session() -> None:
    from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin

    orchestrator, clients, rpc_q, output_q = _build(stages=2, plugin=AuraDuplexPlugin(_encode_audio))
    await _open(orchestrator, rpc_q)
    request_id = next(iter(orchestrator.request_states))
    await _submit(orchestrator, _append_audio())
    request_state = orchestrator.request_states[request_id]
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None

    absorbed = await orchestrator._handle_forward_failure(request_id, 1, request_state, ValueError("stale talker"))
    await _settle(orchestrator)

    assert absorbed is True
    assert request_id not in orchestrator.request_states
    assert SESSION_ID in orchestrator.session_manager.runners
    assert session.state != DuplexSessionState.CLOSED
    types = [message.event.type for message in [output_q.get_nowait() for _ in range(output_q.qsize())]]
    assert "error" in types
    assert "session.expired" not in types
    await orchestrator.session_manager.shutdown()


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


# --------------------------------------------------------------------------- #
# The hierarchy the two surfaces depend on                                    #
# --------------------------------------------------------------------------- #


def test_the_duplex_stack_extends_the_turn_based_one() -> None:
    """One engine serves a session and an ordinary request, so duplex must extend turn-based.

    Pinned because it is the kind of relationship a later tidy-up turns back
    into siblings over a shared base. If it does, a duplex server silently
    stops serving /v1/chat/completions: the chat service needs a real
    EngineClient with generate(), which only the turn-based side provides.
    """
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
    from vllm_omni.engine.orchestrator import Orchestrator
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

    assert issubclass(DuplexOrchestrator, Orchestrator)
    assert issubclass(DuplexOmniEngine, AsyncOmniEngine)
    assert issubclass(DuplexOmni, AsyncOmni)
    # What the chat route actually needs off the engine client.
    assert hasattr(DuplexOmni, "generate")


@pytest.mark.asyncio
async def test_an_unrecognised_message_reaches_the_turn_based_handler() -> None:
    """The duplex orchestrator consumes session messages and passes the rest down.

    Without the fall-through, an ordinary add_request on a duplex engine is
    silently dropped and the chat request that sent it hangs forever.
    """
    orchestrator, _clients, _rpc_q, _output_q = _build()
    handled: list[str] = []

    async def _turn_based(msg: object) -> None:
        handled.append(getattr(msg, "type", ""))

    orchestrator._handle_add_request = _turn_based  # type: ignore[method-assign]
    msg = SimpleNamespace(type="add_request")

    assert await orchestrator._dispatch_message(msg) is True
    assert handled == ["add_request"], "an add_request must reach Orchestrator, not be dropped"


@pytest.mark.asyncio
async def test_a_dead_replica_closes_the_sessions_it_was_serving() -> None:
    """A session outlives its stage request, so replica death has to release the owner.

    ``_handle_dead_replica`` cleaned the stage request but not the runner that
    owned it: the session stayed alive holding an admission slot, and the client
    was told nothing — the request-scoped ``ErrorMessage`` it emits has no
    frontend ``request_states`` entry to land in for a session-owned request, so
    the frontend drops it. Heartbeats would then keep an unusable session
    occupying capacity.
    """
    orchestrator, clients, rpc_q, output_q = _build()
    await _open(orchestrator, rpc_q)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None
    assert orchestrator.session_manager.active_count() == 1

    await orchestrator._handle_dead_replica(0, 0, EngineDeadError("stage 0 replica 0 died"))
    await _settle(orchestrator)

    assert request_id not in orchestrator.request_states
    assert SESSION_ID not in orchestrator.session_manager.runners
    assert session.state == DuplexSessionState.CLOSED
    assert orchestrator.session_manager.active_count() == 0, "the admission slot must come back"
    messages = [output_q.get_nowait() for _ in range(output_q.qsize())]
    types = [getattr(getattr(m, "event", None), "type", type(m).__name__) for m in messages]
    assert types[-1] in {"session.expired", "session.closed"}, f"the client needs a terminal event, got {types}"


@pytest.mark.asyncio
async def test_turn_plugin_processes_multimodal_prompt_before_stage_submission(monkeypatch):
    from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
    from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

    orchestrator, clients, rpc_q, _ = _build()
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    plugin.processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: "audio prompt")
    orchestrator.plugin = orchestrator.session_manager.plugin = plugin
    seen = []

    def process_inputs(**kwargs):
        seen.append(kwargs)
        return build_engine_core_request_from_tokens(
            request_id=kwargs["request_id"],
            prompt={"prompt_token_ids": [1, 2]},
            params=kwargs["params"],
            model_config=orchestrator.stage_pools[0].stage_vllm_config.model_config,
            resumable=kwargs["resumable"],
        )

    monkeypatch.setattr(
        orchestrator, "_get_stage_input_processor", lambda stage_id: SimpleNamespace(process_inputs=process_inputs)
    )
    await orchestrator._dispatch_message(
        OpenDuplexSessionMessage(
            control_id="open-qwen",
            session_id=SESSION_ID,
            session_config=DuplexSessionConfig(model="qwen", modalities=["text", "audio"]),
        )
    )
    assert (await rpc_q.get()).ok
    try:
        await _submit(orchestrator, _append_audio())
        assert not clients[0].add_request_calls
        await _submit(orchestrator, commands.Commit(final=True, create_response=True))
        assert len(seen) == 1
        assert seen[0]["prompt"]["multi_modal_data"]["audio"][0][0].shape == (16000,)
        submitted = clients[0].add_request_calls[0][0]
        assert submitted.resumable is False
        assert not orchestrator.request_states[submitted.request_id].streaming.enabled
    finally:
        await _close(orchestrator, rpc_q)


@pytest.mark.asyncio
async def test_sentence_partial_does_not_legacy_forward_the_full_stage_output() -> None:
    from vllm_omni.engine.duplex.plugin import PartialStageForward

    orchestrator, *_ = _build(stages=4)
    forwarded: list[object] = []

    async def record_forward(req_id, stage_id, output, req_state, **kwargs):
        del req_id, stage_id, req_state, kwargs
        forwarded.append(output)

    async def no_intercept(*args, **kwargs):
        del args, kwargs
        return False

    orchestrator._forward_to_next_stage = record_forward  # type: ignore[method-assign]
    orchestrator._intercept_stage_output = no_intercept  # type: ignore[method-assign]
    orchestrator.async_chunk = True
    orchestrator._stage_receives_async_chunks = lambda stage_id: False  # type: ignore[method-assign]

    full = SimpleNamespace(request_id="r", finished=True, text="FULL")
    chunk = SimpleNamespace(request_id="r", finished=True, text="CHUNK")
    req_state = DuplexOrchestratorRequestState(
        request_id="r",
        final_stage_id=3,
        session_owned=True,
        sampling_params_list=[SimpleNamespace() for _ in range(4)],
    )
    orchestrator.plugin.plan_partial_stage_output = lambda *args, **kwargs: PartialStageForward(  # type: ignore[method-assign]
        output=chunk, is_final_update=True
    )
    orchestrator.plugin.partial_stage_followup = lambda *args, **kwargs: None  # type: ignore[method-assign]

    await orchestrator._route_output(1, 0, full, req_state, None)

    assert forwarded == [chunk]
    assert req_state.skip_legacy_stage_forward is False

    forwarded.clear()
    orchestrator.plugin.plan_partial_stage_output = lambda *args, **kwargs: None  # type: ignore[method-assign]
    await orchestrator._route_output(1, 0, full, req_state, None)

    assert forwarded == [full]


def test_default_plugin_declares_no_draining_stages() -> None:
    plugin = MiniCPMO45DuplexPlugin(_encode_audio)
    assert plugin.draining_stage_ids(stage_count=4) == frozenset()
