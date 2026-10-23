# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``DuplexOrchestrator``: the template seams and the stage port around ``OrchestratorBase``."""

from __future__ import annotations

import asyncio
import struct
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from vllm.outputs import CompletionOutput
from vllm.v1.engine import FinishReason
from vllm.v1.engine.exceptions import EngineDeadError

from tests.engine.test_orchestrator import (
    FakeOutputProcessor,
    FakeRunningCounter,
    FakeStageClient,
    _build_stage_pools,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator, DuplexOrchestratorRequestState
from vllm_omni.engine.messages import AbortRequestMessage, AddCompanionRequestMessage, ShutdownRequestMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState, build_engine_core_request_from_tokens
from vllm_omni.engine.stage_pool import StageUnavailableError
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.outputs import OmniRequestOutput

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
    replicas: int = 1,
    running_counter: FakeRunningCounter | None = None,
    runtime_config: DuplexSessionRuntimeConfig | None = None,
    plugin=None,
    frontend_count: int = 1,
    frontend_index: int = 0,
) -> tuple[DuplexOrchestrator, list[FakeStageClient], asyncio.Queue, asyncio.Queue]:
    clients = [FakeStageClient(stage_type="llm", final_output=index == stages - 1) for index in range(stages)]
    pools = _build_stage_pools(
        [
            [client, *[FakeStageClient(final_output=index == stages - 1) for _ in range(replicas - 1)]]
            for index, client in enumerate(clients)
        ],
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
        plugin=plugin or MiniCPMO45DuplexPlugin(_encode_audio),
        duplex_session_config=runtime_config or DuplexSessionRuntimeConfig(reaper_interval_s=0.01),
        model_config=None,
        frontend_count=frontend_count,
        frontend_index=frontend_index,
    )
    return orchestrator, clients, rpc_q, output_q


def _open_message(
    extra_body: dict[str, object] | None = None,
    *,
    output_buffer: DuplexOutputBuffer | None = None,
    session_id: str = SESSION_ID,
) -> OpenDuplexSessionMessage:
    return OpenDuplexSessionMessage(
        control_id=f"open-{session_id}",
        session_id=session_id,
        output_buffer=output_buffer or DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512),
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


def _audio_request_output(
    request_id: str,
    *,
    samples: int,
    finished: bool,
    text: str = "hello",
    turn_end: bool = False,
) -> OmniRequestOutput:
    multimodal: dict[str, object] = {
        "audio": np.zeros(samples, dtype=np.float32),
        "sr": 24000,
        "meta.duplex_turn_id": np.array([0], dtype=np.int32),
        "meta.duplex_epoch": np.array([0], dtype=np.int32),
    }
    if turn_end:
        multimodal["turn_end"] = True
    return OmniRequestOutput(
        request_id=request_id,
        finished=finished,
        final_output_type="audio",
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop" if finished else None,
            )
        ],
        _multimodal_output=multimodal,
    )


async def _route_raw_codec_output(
    orchestrator: DuplexOrchestrator,
    *,
    stage_id: int,
    request_id: str,
    processed: OmniRequestOutput,
    is_segment_finished: bool,
    finish_reason: FinishReason | None = FinishReason.STOP,
) -> None:
    """Drive one codec poll through raw bookkeeping, the interceptor, and the projector.

    Session-owned requests skip ``_finish_raw_terminal_requests``. The raw
    record these tests lock is ``finished_stage_ids`` from
    ``_apply_raw_terminal_stage_finish``.
    """
    orchestrator.stage_pools[stage_id].output_processor.request_outputs = [processed]
    raw = OmniEngineCoreOutputs(
        outputs=[
            OmniEngineCoreOutput(
                request_id=request_id,
                new_token_ids=[],
                finish_reason=finish_reason,
                is_segment_finished=is_segment_finished,
            )
        ]
    )
    terminal: set[str] = set()
    routed = await orchestrator._process_llm_stage_outputs(stage_id, 0, raw, terminal)
    await orchestrator._handle_processed_outputs(stage_id, 0, routed)
    await orchestrator._finish_raw_terminal_requests(stage_id, 0, terminal)
    await _settle(orchestrator)


async def _drain_events(output_buffer: DuplexOutputBuffer, output_q: asyncio.Queue) -> list[Any]:
    events = [await output_buffer.get() for _ in range(output_buffer.pending_events)]
    while not output_q.empty():
        message = output_q.get_nowait()
        event = getattr(message, "event", message)
        events.append(event)
    return events


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


@pytest.fixture
def ephemeral_submissions(mocker):
    orchestrator, clients, _, _ = _build(stages=2)
    submissions = []
    for request_id in ("first", "second"):
        context = DuplexStageRequestContext(
            request_id=request_id,
            session_id=request_id,
            fence=DuplexFence(request_id),
            stage_id=0,
            final_stage_id=1,
            config_generation=0,
            sampling_params=orchestrator.sampling_defaults(),
        )
        orchestrator.ensure_request(context)
        submissions.append(
            DuplexStageSubmission(
                context=context, prompt={"prompt_token_ids": [1]}, already_submitted=False, resumable=False
            )
        )

    async def build_request(request_id, stage_id, prompt, params, *, resumable):
        return build_engine_core_request_from_tokens(request_id, prompt, params, resumable=resumable)

    builder = mocker.patch.object(orchestrator, "_build_next_stage_request", side_effect=build_request)
    return orchestrator, clients, submissions, builder


@pytest.mark.asyncio
async def test_nonzero_entry_stage_retains_direct_entry_routing(mocker):
    orchestrator, clients, _, _ = _build(stages=4)
    context = DuplexStageRequestContext(
        request_id="camera-only",
        session_id="camera",
        fence=DuplexFence("camera"),
        stage_id=1,
        final_stage_id=3,
        config_generation=0,
        sampling_params=orchestrator.sampling_defaults(),
    )
    orchestrator.ensure_request(context)
    assert orchestrator.request_states[context.request_id].entry_stage_id == 1

    async def build_request(request_id, stage_id, prompt, params, *, resumable):
        return build_engine_core_request_from_tokens(request_id, prompt, params, resumable=resumable)

    mocker.patch.object(orchestrator, "_build_next_stage_request", side_effect=build_request)
    result = await orchestrator.submit(
        DuplexStageSubmission(
            context=context, prompt={"prompt_token_ids": [1]}, already_submitted=False, resumable=False
        )
    )
    assert result.stage_id == 1
    assert not clients[0].add_request_calls and len(clients[1].add_request_calls) == 1
    await orchestrator.cleanup([context.request_id])


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_ephemeral_prewarm_does_not_hold_stage_input_lock(ephemeral_submissions, mocker, outcome):
    orchestrator, clients, submissions, _ = ephemeral_submissions
    orchestrator.async_chunk = True
    prewarming = asyncio.Event()
    release = asyncio.Event()
    clear_cache = mocker.spy(orchestrator, "_clear_stage_input_cache")

    async def prewarm(request_id, request, request_state):
        if request_id == "first":
            prewarming.set()
            await release.wait()
            return outcome != "failure"
        return True

    mocker.patch.object(orchestrator, "_prewarm_async_chunk_stages", side_effect=prewarm)
    first = asyncio.create_task(orchestrator.submit(submissions[0]))
    try:
        await asyncio.wait_for(prewarming.wait(), timeout=2)
        # A different session can finish admission while the first waits downstream.
        second = await asyncio.wait_for(orchestrator.submit(submissions[1]), timeout=2)
        assert second.request_id == "second"
        assert not first.done()
        assert [args[0].request_id for args in clients[0].add_request_calls] == ["first", "second"]
        if outcome == "cancel":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            release.set()
            if outcome == "failure":
                with pytest.raises(RuntimeError, match="async-chunk prewarm failed"):
                    await first
            else:
                assert (await first).request_id == "first"
        # A downstream failure must not invalidate a later session's sender cache.
        clear_cache.assert_not_called()
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_send", [False, True])
async def test_ephemeral_submission_preserves_send_order_and_failure_cleanup(ephemeral_submissions, mocker, fail_send):
    orchestrator, clients, submissions, builder = ephemeral_submissions
    sending = asyncio.Event()
    release = asyncio.Event()
    add_request = clients[0].add_request_async
    clear_cache = mocker.spy(orchestrator, "_clear_stage_input_cache")

    async def send(request, **kwargs):
        if request.request_id == "first":
            sending.set()
            await release.wait()
            if fail_send:
                raise RuntimeError("send failed")
        else:
            assert clear_cache.call_count == int(fail_send)
        await add_request(request, **kwargs)

    mocker.patch.object(clients[0], "add_request_async", side_effect=send)
    first = asyncio.create_task(orchestrator.submit(submissions[0]))
    second = None
    try:
        await asyncio.wait_for(sending.wait(), timeout=2)
        second = asyncio.create_task(orchestrator.submit(submissions[1]))
        await asyncio.sleep(0)
        assert builder.await_count == 1
        assert orchestrator._stage_input_submission_locks[0].locked()
        release.set()
        if fail_send:
            with pytest.raises(RuntimeError, match="send failed"):
                await first
        else:
            await first
        await asyncio.wait_for(second, timeout=2)
        expected = ["second"] if fail_send else ["first", "second"]
        assert [args[0].request_id for args in clients[0].add_request_calls] == expected
    finally:
        tasks = [first] if second is None else [first, second]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    await _open(orchestrator, rpc_q, output_buffer=output_buffer)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    request_state = orchestrator.request_states[request_id]
    while output_buffer.pending_events:
        await output_buffer.get()

    consumed = await orchestrator._intercept_stage_output(1, 0, _tts_output(request_id), request_state, None, None)
    await _settle(orchestrator)

    assert consumed is True
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None and session.active_response_id is not None
    types = [(await output_buffer.get()).type for _ in range(output_buffer.pending_events)]
    assert "response.created" in types and "response.output_audio.delta" in types
    assert output_q.empty()

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
async def test_streaming_segment_stop_keeps_response_open_until_real_terminal(mocker) -> None:
    """Codec segment EOS must not close the duplex response before the real stop."""
    orchestrator, _, rpc_q, output_q = _build(stages=2)
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    await _open(orchestrator, rpc_q, output_buffer=output_buffer)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    request_state = orchestrator.request_states[request_id]
    assert request_state.streaming.enabled is True
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None
    while output_buffer.pending_events:
        await output_buffer.get()
    mocker.patch.object(orchestrator, "_handle_kv_ready_raw_outputs", new_callable=mocker.AsyncMock)

    try:
        await _route_raw_codec_output(
            orchestrator,
            stage_id=1,
            request_id=request_id,
            processed=_audio_request_output(request_id, samples=2400, finished=True, text="seg"),
            is_segment_finished=True,
        )
        after_segment = await _drain_events(output_buffer, output_q)
        assert any(event.type == "response.output_audio.delta" for event in after_segment)
        assert not any(event.type == "response.done" for event in after_segment)
        assert session.active_response_id is not None
        assert request_state.streaming.segment(1).finished is True
        assert request_id in orchestrator.request_states
        assert 1 not in request_state.finished_stage_ids
        assert 1 not in request_state.finished_final_output_stage_ids

        await _route_raw_codec_output(
            orchestrator,
            stage_id=1,
            request_id=request_id,
            processed=_audio_request_output(request_id, samples=7200, finished=False, text="more"),
            is_segment_finished=False,
            finish_reason=None,
        )
        after_more = await _drain_events(output_buffer, output_q)
        assert any(event.type == "response.output_audio.delta" for event in after_more)
        assert not any(event.type == "response.done" for event in after_more)
        assert session.active_response_id is not None

        response_id = session.active_response_id
        await _route_raw_codec_output(
            orchestrator,
            stage_id=1,
            request_id=request_id,
            processed=_audio_request_output(request_id, samples=8400, finished=True, text="end", turn_end=True),
            is_segment_finished=False,
        )
        after_terminal = await _drain_events(output_buffer, output_q)
        done = [event for event in after_terminal if event.type == "response.done"]
        assert len(done) == 1
        assert done[0].response_id == response_id
        assert done[0].status == "completed"
        assert session.active_response_id is None
        assert 1 in request_state.finished_stage_ids
        assert 1 in request_state.finished_final_output_stage_ids
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_non_streaming_terminal_completes_the_response_once(mocker) -> None:
    """A finished codec stop with is_segment_finished=False still ends exactly once."""
    orchestrator, _, rpc_q, output_q = _build(stages=2)
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    await _open(orchestrator, rpc_q, output_buffer=output_buffer)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    request_state = orchestrator.request_states[request_id]
    session = orchestrator.session_manager.get(SESSION_ID)
    assert session is not None
    while output_buffer.pending_events:
        await output_buffer.get()
    mocker.patch.object(orchestrator, "_handle_kv_ready_raw_outputs", new_callable=mocker.AsyncMock)

    try:
        await _route_raw_codec_output(
            orchestrator,
            stage_id=1,
            request_id=request_id,
            processed=_audio_request_output(request_id, samples=2400, finished=True, text="done", turn_end=True),
            is_segment_finished=False,
        )
        events = await _drain_events(output_buffer, output_q)
        done = [event for event in events if event.type == "response.done"]
        assert any(event.type == "response.output_audio.delta" for event in events)
        assert len(done) == 1
        assert done[0].status == "completed"
        assert session.active_response_id is None
        assert request_state.streaming.segment(1).finished is False
        assert 1 in request_state.finished_stage_ids
        assert 1 in request_state.finished_final_output_stage_ids
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_id", [0, 1])
async def test_terminal_stage_error_closes_session_before_another_append(mocker, stage_id) -> None:
    orchestrator, clients, rpc_q, output_q = _build(stages=2)
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    await _open(orchestrator, rpc_q, output_buffer=output_buffer)
    request_id = _stage0_request_id()
    await _submit(orchestrator, _append_audio())
    session = orchestrator.session_manager.get(SESSION_ID)
    reason = "context_length_exceeded: streaming session prompt would grow to 3000 tokens"
    output = OmniEngineCoreOutput(
        request_id=request_id,
        finish_reason=FinishReason.ERROR,
        stop_reason=reason,
        is_segment_finished=False,
        new_token_ids=[],
    )
    mocker.patch.object(orchestrator, "_handle_kv_ready_raw_outputs", new_callable=mocker.AsyncMock)
    mocker.patch.object(
        orchestrator.stage_pools[stage_id], "process_llm_raw_outputs", new_callable=mocker.AsyncMock, return_value=[]
    )
    try:
        await orchestrator._process_llm_stage_outputs(stage_id, 0, OmniEngineCoreOutputs(outputs=[output]), set())
        await _settle(orchestrator)
        assert request_id not in orchestrator.request_states
        assert clients[0].abort_calls == [[request_id]]
        assert SESSION_ID not in orchestrator.session_manager.runners
        assert session.state == DuplexSessionState.CLOSED
        assert orchestrator.session_manager.active_count() == 0
        events = [await output_buffer.get() for _ in range(output_buffer.pending_events)]
        events.extend(output_q.get_nowait().event for _ in range(output_q.qsize()))
        types = [event.type for event in events]
        assert types.index("error") < types.index("session.expired")
        assert types[-1] == "session.expired"
        # A request-local failure must release admission without evicting the replica.
        result = await _open(orchestrator, rpc_q)
        assert result.ok
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin_type,close_session", [(MiniCPMO45DuplexPlugin, True), (Qwen3OmniDuplexPlugin, False)])
async def test_terminal_stage_error_preserves_plugin_cleanup_policy(mocker, plugin_type, close_session) -> None:
    orchestrator, _, _, _ = _build(plugin=plugin_type(_encode_audio))
    request_state = DuplexOrchestratorRequestState(request_id="failed-request", session_owned=True)
    output = OmniEngineCoreOutput(
        request_id=request_state.request_id,
        finish_reason=FinishReason.ERROR,
        stop_reason="context_length_exceeded",
        new_token_ids=[],
    )
    cleanup = mocker.patch.object(orchestrator, "_cleanup_request_ids", new_callable=mocker.AsyncMock)
    try:
        await orchestrator._report_duplex_session_request_error(0, 0, output, request_state)
        cleanup.assert_awaited_once_with([request_state.request_id], abort=True, release_owners=close_session)
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_qwen_terminal_stage_error_allows_a_subsequent_turn(mocker) -> None:
    from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    render_prompt = mocker.Mock(return_value="audio prompt")
    plugin.processor = SimpleNamespace(apply_chat_template=render_prompt)
    orchestrator, clients, rpc_q, output_q = _build(plugin=plugin)
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)

    def process_inputs(**kwargs):
        return build_engine_core_request_from_tokens(
            request_id=kwargs["request_id"],
            prompt={"prompt_token_ids": [1, 2]},
            params=kwargs["params"],
            model_config=orchestrator.stage_pools[0].stage_vllm_config.model_config,
            resumable=kwargs["resumable"],
        )

    mocker.patch.object(
        orchestrator, "_get_stage_input_processor", return_value=SimpleNamespace(process_inputs=process_inputs)
    )
    assert (await _open(orchestrator, rpc_q, extra_body={"auto_response": False}, output_buffer=output_buffer)).ok
    session = orchestrator.session_manager.get(SESSION_ID)
    try:
        await _submit(orchestrator, _append_audio())
        await _submit(orchestrator, commands.Commit(final=True, create_response=True))
        first_request = clients[0].add_request_calls[0][0]
        failed_response_id = session.active_response_id
        assert failed_response_id is not None

        await orchestrator._process_llm_stage_outputs(
            0,
            0,
            OmniEngineCoreOutputs(
                outputs=[
                    OmniEngineCoreOutput(
                        request_id=first_request.request_id,
                        finish_reason=FinishReason.ERROR,
                        stop_reason="context_length_exceeded",
                        new_token_ids=[],
                    )
                ]
            ),
            set(),
        )
        await _settle(orchestrator)
        assert clients[0].abort_calls == [[first_request.request_id]]
        assert first_request.request_id not in orchestrator.request_states
        assert session.state == DuplexSessionState.OPEN
        assert session.active_response_id is None
        assert orchestrator.session_manager.get(SESSION_ID) is session

        await _submit(orchestrator, _append_audio())
        await _submit(orchestrator, commands.Commit(final=True, create_response=True))
        assert len(clients[0].add_request_calls) == 2
        second_request = clients[0].add_request_calls[1][0]
        assert second_request.request_id != first_request.request_id
        assert second_request.resumable is False
        # Each turn rebuilds the prompt, including the session instructions.
        assert render_prompt.call_count == 2
        assert render_prompt.call_args.args[0][0] == {"role": "system", "content": "public instructions"}
        response_id = session.active_response_id
        assert response_id is not None and response_id != failed_response_id
        output = SimpleNamespace(
            request_id=second_request.request_id,
            finished=True,
            outputs=[SimpleNamespace(text="next turn succeeded", token_ids=[], multimodal_output={})],
            multimodal_output={},
        )
        assert await orchestrator._intercept_stage_output(
            0, 0, output, orchestrator.request_states[second_request.request_id], None, None
        )
        await _settle(orchestrator)
        events = [await output_buffer.get() for _ in range(output_buffer.pending_events)]
        events.extend(output_q.get_nowait().event for _ in range(output_q.qsize()))
        done = [event for event in events if event.type == "response.done"]
        assert [(event.response_id, event.status) for event in done] == [
            (failed_response_id, "failed"),
            (response_id, "completed"),
        ]
        assert any(event.type == "error" for event in events)
        assert not any(event.type in {"session.closed", "session.expired"} for event in events)
        assert session.state == DuplexSessionState.OPEN
    finally:
        await _close(orchestrator, rpc_q)


@pytest.mark.asyncio
async def test_forward_failure_closes_the_owning_session() -> None:
    orchestrator, clients, rpc_q, output_q = _build(stages=2)
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    await _open(orchestrator, rpc_q, output_buffer=output_buffer)
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
    types = [(await output_buffer.get()).type for _ in range(output_buffer.pending_events)]
    types.extend(output_q.get_nowait().event.type for _ in range(output_q.qsize()))
    # The stage failure is reported before the session expires, never after.
    assert types.index("error") < types.index("session.expired")
    assert types[-1] == "session.expired"


@pytest.mark.asyncio
async def test_aura_forward_failure_keeps_the_session() -> None:
    from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin

    orchestrator, clients, rpc_q, output_q = _build(stages=2, plugin=AuraDuplexPlugin(_encode_audio))
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    await _open(orchestrator, rpc_q, output_buffer=output_buffer)
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
    types = [(await output_buffer.get()).type for _ in range(output_buffer.pending_events)]
    types.extend(output_q.get_nowait().event.type for _ in range(output_q.qsize()))
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
            output_buffer=DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512),
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


def _placement_config(**kwargs) -> DuplexSessionRuntimeConfig:
    return DuplexSessionRuntimeConfig(
        replica_policy="least_sessions", max_sessions=8, max_sessions_per_replica=2, **kwargs
    )


@pytest.mark.asyncio
async def test_payload_sender_info_binds_from_explicit_session_state(monkeypatch) -> None:
    orchestrator, _, rpc_q, _ = _build(replicas=2, runtime_config=_placement_config(replica_groups=[{0: 1}, {0: 0}]))
    pool = orchestrator.stage_pools[0]
    for replica, client in enumerate(pool.clients):
        client.get_payload_sender_info = lambda r=replica: {"replica": r}
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        request_id = _stage0_request_id()
        state = orchestrator.request_states[request_id]
        assert pool.get_bound_replica_id(request_id) is None
        with monkeypatch.context() as patch:
            patch.delattr(orchestrator, "request_states")
            assert orchestrator._build_payload_sender_info(0, request_id=request_id) == {"replica": 0}
            assert pool.get_bound_replica_id(request_id) is None
            sender_info = orchestrator._build_payload_sender_info(0, request_id=request_id, req_state=state)
        assert sender_info == {"replica": 1}
        assert pool.get_bound_replica_id(request_id) == 1
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_session_group_pins_initial_update_forward_prewarm_and_new_epoch(monkeypatch) -> None:
    config = _placement_config(replica_groups=[{0: 1, 1: 0, 2: 1}, {0: 0, 1: 1, 2: 0}])
    orchestrator, _, rpc_q, _ = _build(stages=3, replicas=2, runtime_config=config)
    pools = orchestrator.stage_pools
    orchestrator.async_chunk = True
    monkeypatch.setattr(orchestrator, "_stage_receives_async_chunks", lambda stage: stage == 2)
    for stage, pool in enumerate(pools):
        for replica, client in enumerate(pool.clients):
            client.next_inputs = [{"prompt_token_ids": [1, 2]}]
            client.get_payload_sender_info = lambda s=stage, r=replica: {"stage": s, "replica": r}
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        placement = orchestrator.replica_placement
        before = placement.snapshot()["reserved"]
        # Opening reserves capacity without adding session placeholders to a pool.
        assert all(not pool._request_bindings for pool in pools)
        await _submit(orchestrator, _append_audio())
        request_id = _stage0_request_id()
        state = orchestrator.request_states[request_id]
        codec_request = pools[2].clients[1].add_request_calls[0][0]
        # Stage 1 has not submitted yet, but its sender endpoint is already pinned.
        assert not pools[1].clients[0].add_request_calls
        assert codec_request.payload_sender_info == {"stage": 1, "replica": 0}
        await orchestrator._forward_to_next_stage(
            request_id, 0, SimpleNamespace(request_id=request_id), state, is_streaming_session=True
        )
        await _submit(orchestrator, _append_audio())
        assert len(pools[0].clients[1].add_request_calls) == 2
        assert pools[1].clients[0].add_request_calls
        for stage, replica in {0: 1, 1: 0, 2: 1}.items():
            assert pools[stage].get_bound_replica_id(request_id) == replica
            assert not pools[stage].clients[1 - replica].add_request_calls
        await _submit(orchestrator, commands.BargeIn())
        assert request_id not in orchestrator.request_states
        assert placement.snapshot()["reserved"] == before
        await _submit(orchestrator, _append_audio())
        new_id = _stage0_request_id(epoch=1)
        assert pools[0].get_bound_replica_id(new_id) == 1
        assert pools[2].get_bound_replica_id(new_id) == 1
        await _close(orchestrator, rpc_q)
        assert all(value == 0 for value in placement.snapshot()["reserved"].values())
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_turn_requests_keep_the_session_replica_before_prompt_processing(monkeypatch) -> None:
    from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    plugin.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "audio prompt")
    orchestrator, _, rpc_q, _ = _build(
        replicas=2,
        runtime_config=_placement_config(replica_groups=[{0: 1}, {0: 0}]),
        plugin=plugin,
    )
    pool = orchestrator.stage_pools[0]
    processed_ids = []

    def process_inputs(**kwargs):
        request_id = kwargs["request_id"]
        assert pool.get_bound_replica_id(request_id) == 1
        assert orchestrator._stage_input_submission_locks[0].locked()
        processed_ids.append(request_id)
        return build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt={"prompt_token_ids": [1, 2]},
            params=kwargs["params"],
            model_config=pool.stage_vllm_config.model_config,
            resumable=False,
        )

    monkeypatch.setattr(
        orchestrator, "_get_stage_input_processor", lambda stage: SimpleNamespace(process_inputs=process_inputs)
    )
    try:
        assert (await _open(orchestrator, rpc_q, extra_body={"auto_response": False})).ok
        session = orchestrator.session_manager.get(SESSION_ID)
        for _ in range(2):
            await _submit(orchestrator, _append_audio())
            await _submit(orchestrator, commands.Commit(final=True, create_response=True))
            request_id = pool.clients[1].add_request_calls[-1][0].request_id
            state = orchestrator.request_states[request_id]
            output = SimpleNamespace(
                request_id=request_id,
                finished=True,
                outputs=[SimpleNamespace(text="done", token_ids=[], multimodal_output={})],
                multimodal_output={},
            )
            await orchestrator._intercept_stage_output(0, 1, output, state, None, None)
            await _settle(orchestrator)
            assert session.active_response_id is None
        assert len(set(processed_ids)) == 2
        assert not pool.clients[0].add_request_calls
        assert orchestrator.replica_placement.snapshot()["sessions"] == {SESSION_ID: 0}
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("preselected", [False, True])
async def test_session_replica_scopes_multimodal_cache_before_processing(monkeypatch, preselected) -> None:
    from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

    orchestrator, _, rpc_q, _ = _build(replicas=2, runtime_config=_placement_config())
    processed = []
    scope_calls = []
    loop_thread_id = threading.get_ident()
    scope_prompt = orchestrator._scope_stage_mm_cache_to_replica

    def scope_inputs(stage_id, prompt, replica_id):
        assert threading.get_ident() != loop_thread_id
        scope_calls.append((stage_id, replica_id))
        return scope_prompt(stage_id, prompt, replica_id)

    def process_inputs(**kwargs):
        assert threading.get_ident() != loop_thread_id
        processed.append(kwargs["prompt"])
        return build_engine_core_request_from_tokens(
            request_id=kwargs["request_id"], prompt={"prompt_token_ids": [1]}, params=kwargs["params"]
        )

    monkeypatch.setattr(
        orchestrator, "_get_stage_input_processor", lambda stage: SimpleNamespace(process_inputs=process_inputs)
    )
    monkeypatch.setattr(orchestrator, "_scope_stage_mm_cache_to_replica", scope_inputs)
    prompt = {"prompt": "image", "multi_modal_data": {"image": object()}, "multi_modal_uuids": {"image": ["same"]}}
    try:
        for session_id in (SESSION_ID, "other"):
            assert (await _open(orchestrator, rpc_q, session_id=session_id)).ok
            request_id = duplex_resource_request_id(DuplexFence(session_id), "stage0")
            state = orchestrator.request_states[request_id]
            replica_id = orchestrator._session_route(0, state)
            orchestrator.stage_pools[0].bind_request_to_replica(request_id, replica_id)
            kwargs = {"replica_id": replica_id} if preselected else {}
            await orchestrator._build_next_stage_request(request_id, 0, prompt, state.sampling_params_list[0], **kwargs)
        assert scope_calls == [(0, 0), (0, 1)]
        assert [item["multi_modal_uuids"]["image"] for item in processed] == [
            ["stage0:rep0:same"],
            ["stage0:rep1:same"],
        ]
        assert prompt["multi_modal_uuids"] == {"image": ["same"]}
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_derived_companion_uses_session_pin_even_without_parent_request_binding() -> None:
    from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

    orchestrator, _, rpc_q, _ = _build(replicas=2, runtime_config=_placement_config(replica_groups=[{0: 1}, {0: 0}]))
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        parent_id = _stage0_request_id()
        state = orchestrator.request_states[parent_id]
        prompt = build_engine_core_request_from_tokens(
            request_id="companion", prompt={"prompt_token_ids": [1]}, params=state.sampling_params_list[0]
        )
        await orchestrator._handle_add_companion(
            AddCompanionRequestMessage(
                companion_id="companion",
                parent_id=parent_id,
                role="uncond",
                prompt=prompt,
                companion_prompt_text=None,
                sampling_params_list=state.sampling_params_list,
            )
        )
        pool = orchestrator.stage_pools[0]
        assert pool.get_bound_replica_id("companion") == 1
        assert not pool.clients[0].add_request_calls
        assert pool.clients[1].add_request_calls[0][0] is prompt
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_stage", [0, 1])
async def test_failed_initial_submit_evicts_session_replica_after_binding_rollback(monkeypatch, failing_stage) -> None:
    orchestrator, _, rpc_q, _ = _build(stages=2, replicas=2, runtime_config=_placement_config())
    orchestrator.async_chunk = True
    monkeypatch.setattr(orchestrator, "_stage_receives_async_chunks", lambda stage: True)

    async def fail_add(*args, **kwargs):
        raise EngineDeadError("failed initial ADD")

    monkeypatch.setattr(orchestrator.stage_pools[failing_stage].clients[0], "add_request_async", fail_add)
    try:
        for session_id in (SESSION_ID, "healthy", "idle"):
            assert (await _open(orchestrator, rpc_q, session_id=session_id)).ok
        await _submit(orchestrator, _append_audio())
        manager = orchestrator.session_manager
        await asyncio.wait_for(asyncio.gather(*tuple(manager._dispatched_control_tasks)), timeout=1)
        await asyncio.wait_for(manager.reap_expired(), timeout=1)
        assert not manager._closing
        assert orchestrator.replica_placement.snapshot()["sessions"] == {"healthy": 1}
        assert set(orchestrator.session_manager.runners) == {"healthy"}
        assert not orchestrator.stage_pools[failing_stage].is_replica_available(0)
        assert (failing_stage, 0) in orchestrator.replica_placement.snapshot()["unavailable"]
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_replica_failure_closes_active_and_idle_owners_only() -> None:
    orchestrator, _, rpc_q, output_q = _build(stages=2, replicas=2, runtime_config=_placement_config())
    buffers = {}
    try:
        for session_id in (SESSION_ID, "healthy", "idle"):
            buffers[session_id] = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
            assert (await _open(orchestrator, rpc_q, session_id=session_id, output_buffer=buffers[session_id])).ok
        await _submit(orchestrator, _append_audio())
        # No owner has submitted anything to stage 1; request-only failure scans miss them.
        await orchestrator._handle_dead_replica(1, 0, EngineDeadError("test replica death"))
        await _settle(orchestrator)
        manager = orchestrator.session_manager
        await asyncio.wait_for(asyncio.gather(*tuple(manager._dispatched_control_tasks)), timeout=1)
        await asyncio.wait_for(manager.reap_expired(), timeout=1)
        assert not manager._closing
        assert orchestrator.replica_placement.snapshot()["sessions"] == {"healthy": 1}
        assert set(orchestrator.session_manager.runners) == {"healthy"}
        terminal_events = [output_q.get_nowait().event for _ in range(output_q.qsize())]
        for session_id in (SESSION_ID, "idle"):
            events = [await buffers[session_id].get() for _ in range(buffers[session_id].pending_events)]
            events.extend(event for event in terminal_events if event.session_id == session_id)
            errors = [event for event in events if event.type == "error"]
            assert any(event.code == "replica_unavailable" for event in errors)
            assert events[-1].type == "session.expired"
        healthy_events = [await buffers["healthy"].get() for _ in range(buffers["healthy"].pending_events)]
        assert not any(event.type in {"error", "session.expired"} for event in healthy_events)
        assert (await _open(orchestrator, rpc_q, session_id="replacement")).ok
        assert orchestrator.replica_placement.route("replacement", 1) == 1
        result = await _open(orchestrator, rpc_q, session_id="overflow")
        assert not result.ok and result.error_code == "resource_exhausted"
        assert "replica_capacity_exhausted" in result.error_message
        # Surviving sessions can still submit through their original group.
        session = orchestrator.session_manager.get("healthy")
        context = orchestrator.session_manager.ensure_stage_request(session, stage_id=0)
        state = orchestrator.request_states[context.request_id]
        assert orchestrator._session_route(0, state) == 1
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_legacy_session_open_keeps_request_round_robin_and_has_no_placement() -> None:
    orchestrator, _, rpc_q, _ = _build(replicas=2)
    try:
        assert orchestrator.replica_placement is None
        assert (await _open(orchestrator, rpc_q)).ok
        pool = orchestrator.stage_pools[0]
        assert not pool._request_bindings
        assert pool.select_replica_id("ordinary-first") == 0
        await _submit(orchestrator, _append_audio())
        assert pool.get_bound_replica_id(_stage0_request_id()) == 1
        assert pool.select_replica_id("ordinary-second") == 0
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.parametrize("distributed_hook", ["hub", "lb", "membership"])
def test_least_sessions_rejects_distributed_startup(distributed_hook) -> None:
    pools = _build_stage_pools([[FakeStageClient()]], stage_vllm_configs=_stage_configs(1))
    kwargs = {}
    if distributed_hook == "hub":
        pools[0].attach_hub(SimpleNamespace())
    elif distributed_hook == "lb":
        pools[0].attach_load_balancer(SimpleNamespace())
    else:
        kwargs["membership_controller"] = SimpleNamespace()
    with pytest.raises(ValueError, match="does not support distributed"):
        DuplexOrchestrator(
            request_async_queue=asyncio.Queue(),
            output_async_queue=asyncio.Queue(),
            rpc_async_queue=asyncio.Queue(),
            stage_pools=pools,
            plugin=MiniCPMO45DuplexPlugin(_encode_audio),
            duplex_session_config=_placement_config(),
            **kwargs,
        )


def test_orchestrator_validates_groups_against_actual_stage_pools() -> None:
    with pytest.raises(ValueError, match="must cover every replica"):
        _build(stages=2, replicas=2, runtime_config=_placement_config(replica_groups=[{0: 0, 1: 0}]))


@pytest.mark.asyncio
@pytest.mark.parametrize("frontend_index", range(4))
async def test_frontend_reserves_only_owned_groups_with_local_admission_budget(frontend_index) -> None:
    config = DuplexSessionRuntimeConfig(replica_policy="least_sessions", max_sessions=11, max_sessions_per_replica=2)
    orchestrator, _, rpc_q, _ = _build(
        stages=2, replicas=8, runtime_config=config, frontend_count=4, frontend_index=frontend_index
    )
    owned = {frontend_index, frontend_index + 4}
    limit = 3 if frontend_index < 3 else 2
    try:
        assert orchestrator.session_manager.admission_limit == limit
        assert config.max_sessions == 11 and config.worker_max_sessions == 2
        assert all(pool.num_replicas == 8 for pool in orchestrator.stage_pools), "all replicas remain attached"
        for index in range(limit):
            session_id = f"s{index}"
            assert (await _open(orchestrator, rpc_q, session_id=session_id)).ok
            for stage in range(2):
                assert orchestrator.replica_placement.route(session_id, stage) in owned
        rejected = await _open(orchestrator, rpc_q, session_id="full")
        assert rejected.error_code == "resource_exhausted" and rejected.error_retryable
        assert f"limit={limit}" in rejected.error_message
        # Death notifications arrive on non-owner connections too.
        await orchestrator._handle_dead_replica(1, (frontend_index + 1) % 4, EngineDeadError("non-owner"))
        assert len(orchestrator.session_manager.runners) == limit
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_multi_frontend_initial_update_and_prewarm_use_owned_replica(monkeypatch) -> None:
    orchestrator, _, rpc_q, _ = _build(
        stages=2, replicas=4, runtime_config=_placement_config(), frontend_count=2, frontend_index=1
    )
    orchestrator.async_chunk = True
    monkeypatch.setattr(orchestrator, "_stage_receives_async_chunks", lambda stage: True)
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        await _submit(orchestrator, _append_audio())
        await _submit(orchestrator, _append_audio())
        for pool in orchestrator.stage_pools:
            assert pool.get_bound_replica_id(_stage0_request_id()) == 1
            assert pool.clients[1].add_request_calls
            assert all(not pool.clients[index].add_request_calls for index in (0, 2, 3))
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_frontends_emit_reserved_metrics_only_for_owned_replicas() -> None:
    gauge = {}
    writes = []

    def set_reserved(stage, replica, count):
        gauge[stage, replica] = count
        writes.append((stage, replica))

    frontends = [
        _build(stages=2, replicas=2, runtime_config=_placement_config(), frontend_count=2, frontend_index=index)
        for index in range(2)
    ]
    try:
        for index, (orchestrator, _, rpc_q, _) in enumerate(frontends):
            orchestrator._prom_metrics = SimpleNamespace(set_duplex_reserved_sessions=set_reserved)
            assert (await _open(orchestrator, rpc_q, session_id=f"owner-{index}")).ok
            assert writes[-2:] == [(0, index), (1, index)]
        first, _, rpc_q, _ = frontends[0]
        await first.session_manager.close(CloseDuplexSessionMessage(control_id="close", session_id="owner-0"))
        assert (await rpc_q.get()).ok
        assert gauge == {(0, 0): 0, (1, 0): 0, (0, 1): 1, (1, 1): 1}
    finally:
        for orchestrator, _, _, _ in frontends:
            await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["legacy", "least_sessions"])
async def test_single_frontend_keeps_global_admission_limit(policy) -> None:
    config = DuplexSessionRuntimeConfig(replica_policy=policy, max_sessions=11, max_sessions_per_replica=2)
    orchestrator, _, _, _ = _build(runtime_config=config)
    try:
        assert orchestrator.session_manager.admission_limit == 11
        assert config.worker_max_sessions == (2 if policy == "least_sessions" else 11)
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_session_route_query_never_changes_request_bindings() -> None:
    orchestrator, _, rpc_q, _ = _build(replicas=2, runtime_config=_placement_config(replica_groups=[{0: 1}, {0: 0}]))
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        request_id = _stage0_request_id()
        state = orchestrator.request_states[request_id]
        pool = orchestrator.stage_pools[0]
        for _ in range(2):
            assert orchestrator._session_route(0, state) == 1
            assert pool.get_bound_replica_id(request_id) is None
        # Binding conflicts belong to the explicit bind, not the route query.
        pool.bind_request_to_replica(request_id, 0)
        assert orchestrator._session_route(0, state) == 1
        assert pool.get_bound_replica_id(request_id) == 0
        with pytest.raises(StageUnavailableError, match="conflicting"):
            pool.bind_request_to_replica(request_id, 1)
        pool.release_binding(request_id)
        pool.evict_replica(1)
        with pytest.raises(StageUnavailableError, match="unavailable"):
            orchestrator._session_route(0, state)
        assert not pool._request_bindings
        assert orchestrator._session_route(0, OrchestratorRequestState(request_id="ordinary")) is None
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_request_id", [None, "req-x"])
async def test_dead_replica_forwards_failed_request_id(monkeypatch, failed_request_id) -> None:
    orchestrator, _, _, _ = _build(runtime_config=_placement_config())
    error = EngineDeadError("replica died")
    reported = []

    async def record_dead(self, stage_id, replica_id, error, *, failed_request_id=None):
        reported.append((self, stage_id, replica_id, error, failed_request_id))

    monkeypatch.setattr(Orchestrator, "_handle_dead_replica", record_dead)
    try:
        kwargs = {} if failed_request_id is None else {"failed_request_id": failed_request_id}
        await orchestrator._handle_dead_replica(0, 0, error, **kwargs)
        assert reported == [(orchestrator, 0, 0, error, failed_request_id)]
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_dispatch_failure_forwards_failed_request_id_to_base(monkeypatch) -> None:
    orchestrator, clients, _, _ = _build()
    pool = orchestrator.stage_pools[0]
    request_id = "req-x"
    state = OrchestratorRequestState(request_id=request_id, sampling_params_list=list(orchestrator.sampling_defaults()))
    orchestrator.request_states[request_id] = state
    request = build_engine_core_request_from_tokens(
        request_id=request_id, prompt={"prompt_token_ids": [1]}, params=state.sampling_params_list[0]
    )
    error = EngineDeadError("replica died during ADD")
    reported = []

    async def fail_send(*args, **kwargs):
        raise error

    async def record_dead(self, stage_id, replica_id, error, *, failed_request_id=None):
        reported.append((self, stage_id, replica_id, error, failed_request_id))

    monkeypatch.setattr(clients[0], "add_request_async", fail_send)
    # A dead worker cannot acknowledge abort, so the failed submit retains its
    # binding and the dispatch handler can attribute and evict the dead replica.
    monkeypatch.setattr(clients[0], "abort_requests_async", fail_send)
    monkeypatch.setattr(Orchestrator, "_handle_dead_replica", record_dead)
    try:
        submitted = await orchestrator._dispatch_or_fail_request(
            lambda: pool.submit_initial(request_id, state, request),
            req_id=request_id,
            stage_id=0,
            operation="test ADD",
        )
        assert not submitted
        assert reported == [(orchestrator, 0, 0, error, request_id)]
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_dispatch_failure_queries_session_route_without_rebinding(monkeypatch) -> None:
    from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

    orchestrator, _, rpc_q, _ = _build(replicas=2, runtime_config=_placement_config())
    pool = orchestrator.stage_pools[0]
    reported = []

    async def fail_add(*args, **kwargs):
        raise EngineDeadError("initial ADD failed")

    async def record_dead(stage_id, replica_id, error):
        assert not pool._request_bindings, "the error path must not recreate the rolled-back binding"
        reported.append((stage_id, replica_id))

    monkeypatch.setattr(pool.clients[0], "add_request_async", fail_add)
    monkeypatch.setattr(orchestrator, "_handle_dead_replica", record_dead)
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        request_id = _stage0_request_id()
        state = orchestrator.request_states[request_id]
        request = build_engine_core_request_from_tokens(
            request_id=request_id, prompt={"prompt_token_ids": [1]}, params=state.sampling_params_list[0]
        )
        submitted = await orchestrator._dispatch_or_fail_request(
            lambda: pool.submit_initial(request_id, state, request, pinned_replica_id=0),
            req_id=request_id,
            stage_id=0,
            operation="test pinned ADD",
        )
        assert not submitted
        assert reported == [(0, 0)]
        assert not pool._request_bindings
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat_during_cleanup", [True, False])
@pytest.mark.parametrize("failing_stage", [0, 1])
async def test_repeated_replica_failure_reports_session_error_and_close_once(
    monkeypatch, mocker, repeat_during_cleanup, failing_stage
) -> None:
    orchestrator, _, rpc_q, output_q = _build(stages=2, replicas=2, runtime_config=_placement_config())
    manager = orchestrator.session_manager
    placement = orchestrator.replica_placement
    dead_client = orchestrator.stage_pools[failing_stage].clients[0]
    cleanup_started = asyncio.Event()
    cleanup_gate = asyncio.Event()
    cleanup = orchestrator.cleanup
    buffers = {}
    expires = {}

    async def blocked_cleanup(request_ids, *, abort=False):
        cleanup_started.set()
        await cleanup_gate.wait()
        await cleanup(request_ids, abort=abort)

    monkeypatch.setattr(orchestrator, "cleanup", blocked_cleanup)
    try:
        for session_id in (SESSION_ID, "healthy", "idle"):
            buffers[session_id] = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
            assert (await _open(orchestrator, rpc_q, session_id=session_id, output_buffer=buffers[session_id])).ok
            expires[session_id] = mocker.spy(manager.runners[session_id], "expire")
        await _submit(orchestrator, _append_audio())
        error = EngineDeadError("replica reported dead twice")
        await orchestrator._handle_dead_replica(failing_stage, 0, error)
        await asyncio.wait_for(cleanup_started.wait(), timeout=3)
        reserved = placement.snapshot()["reserved"]
        if repeat_during_cleanup:
            await orchestrator._handle_dead_replica(failing_stage, 0, error)
            assert placement.snapshot()["reserved"] == reserved
        cleanup_gate.set()
        await asyncio.wait_for(asyncio.gather(*tuple(manager._dispatched_control_tasks)), timeout=3)
        if not repeat_during_cleanup:
            await orchestrator._handle_dead_replica(failing_stage, 0, error)
        assert dead_client.shutdown_calls == 1
        assert orchestrator.stage_pools[failing_stage].clients[0] is None
        assert not manager._closing
        assert placement.snapshot()["sessions"] == {"healthy": 1}
        assert set(placement.snapshot()["reserved"].values()) == {0, 1}
        messages = [output_q.get_nowait() for _ in range(output_q.qsize())]
        terminal_events = [message.event for message in messages if hasattr(message, "event")]
        for session_id in (SESSION_ID, "idle"):
            events = [await buffers[session_id].get() for _ in range(buffers[session_id].pending_events)]
            events.extend(event for event in terminal_events if event.session_id == session_id)
            assert [event.code for event in events if event.type == "error"] == ["replica_unavailable"]
            assert sum(event.type in {"session.closed", "session.expired"} for event in events) == 1
            assert expires[session_id].await_count == 1
        assert expires["healthy"].await_count == 0
        assert manager.get("healthy") is not None
    finally:
        cleanup_gate.set()
        await manager.shutdown()
