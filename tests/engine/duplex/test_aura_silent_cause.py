# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Spoken AURA turns that must reach Talker instead of going silent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexStageRequestContext,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex.session.context import DuplexRunState, DuplexSessionTasks
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.model_channel import ModelChannel
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestratorRequestState
from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import AuraDataPlaneSession
from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin
from vllm_omni.request import OmniStreamingUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SPOKEN = "看到了，现在画面里有两个人正对着镜头说话。"


class _StagePort:
    def __init__(self) -> None:
        self.stage_count = 4
        self.cleanups: list[tuple[list[str], bool]] = []

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        self.cleanups.append((list(request_ids), abort))

    async def submit(self, submission):
        return DuplexStageSubmissionResult(
            request_id=submission.context.request_id,
            stage_id=submission.context.stage_id,
            replica_id=0,
        )

    def ensure_request(self, context) -> None:
        del context


class _Manager:
    stage_request_id = staticmethod(DuplexSessionManager.stage_request_id)

    def ensure_stage_request(self, session, *, stage_id: int, fence, final_stage_id: int | None = None):
        request_id = self.stage_request_id(
            fence,
            stage_id=stage_id,
            resumable=session.capabilities.supports_core_resumable_request,
        )
        session.reserve_stage_request(stage_id, request_id, fence=fence)
        return DuplexStageRequestContext(
            request_id=request_id,
            session_id=session.session_id,
            fence=fence,
            stage_id=stage_id,
            final_stage_id=3 if final_stage_id is None else final_stage_id,
            config_generation=session.config_generation,
            sampling_params=(SamplingParams(max_tokens=8),),
            session_config=session.config.as_dict(),
            runtime_config={},
        )


def _channel(session: DuplexEngineSession, port: _StagePort, run: DuplexRunState) -> ModelChannel:
    plugin = SimpleNamespace(
        draining_stage_ids=lambda *, stage_count: frozenset({2, 3}) if stage_count >= 4 else frozenset(),
        input_stage_id=lambda payload: 0,
        prepare_prompt_config=lambda config, **kwargs: config,
        extra_prompt_config=lambda runtime_config: {},
        runtime_patch_for_submitted_prompt=lambda prompt: None,
        commit_append_plan=lambda request_id: None,
        prepare_append_plan=lambda **kwargs: _plan(),
        data_plane=SimpleNamespace(close_stream=lambda request_id: None),
    )

    async def prepare_append_plan(**kwargs):
        del kwargs
        return DuplexAppendPlan(prompt={"prompt": ""})

    plugin.prepare_append_plan = prepare_append_plan
    ctx = SimpleNamespace(
        session=session,
        model_state=SimpleNamespace(),
        plugin=plugin,
        stage_port=port,
        manager=_Manager(),
        tasks=DuplexSessionTasks(),
        run=run,
        services=SimpleNamespace(),
        history_calibration=None,
    )
    return ModelChannel(
        ctx,
        SimpleNamespace(emit=lambda payload: None),
        close_from_runtime=lambda reason: _done(),
        schedule_silence_continuation=lambda *args, **kwargs: _done(),
        abort_request=lambda *args, **kwargs: _done(),
    )


async def _plan():
    return DuplexAppendPlan(prompt={"prompt": ""})


async def _done():
    return None


def _ephemeral_session() -> DuplexEngineSession:
    return DuplexEngineSession(
        session_id="duplex-cause",
        config=DuplexSessionConfig(model="aurateam/AURA", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(
            supports_core_resumable_request=False,
            supports_concurrent_turn_requests=True,
        ),
        num_stages=4,
    )


def _opened_request_id(result: dict[str, object] | None) -> str:
    assert result is not None
    stage_results = result["stage_results"]
    assert isinstance(stage_results, list)
    opened = stage_results[0]["result"]["request_id"]
    assert isinstance(opened, str)
    return opened


def _bind_stage0(session: DuplexEngineSession) -> str:
    request_id = DuplexSessionManager.stage_request_id(session.fence, stage_id=0, resumable=False)
    assert request_id.endswith("stage0-turn0")
    session.bind_stage_request(0, request_id, fence=session.fence)
    assert session.stage_request_submitted(0, request_id)
    return request_id


def _bind_inflight_talker(session: DuplexEngineSession) -> str:
    """16:41 shape: one ephemeral id, Stage0 already submitted, Talker on that id."""
    request_id = _bind_stage0(session)
    for stage_id in (2, 3):
        session.bind_stage_request(stage_id, request_id, fence=session.fence)
    return request_id


@pytest.mark.asyncio
async def test_vision_follow_aborts_before_stage2_is_bound() -> None:
    """Stage0 is in flight, but Talker is not bound and the release flag is still false.

    14:55 lands here: aura2tts is inside the submit await, so Stage2 is not in
    request_resources yet. The next commit aborts the shared ephemeral id.
    """
    session = _ephemeral_session()
    request_id = _bind_stage0(session)
    assert not any(stage_id in (2, 3) for stage_id, _ in session.request_resources)
    port = _StagePort()
    run = DuplexRunState(concurrent_turn_requests_released=False)
    channel = _channel(session, port, run)

    await channel._append_via_data_plane(
        {"is_speech": False},
        final=True,
        operation_id="vision-before-bind",
        expected_epoch=None,
        input_audio_part=None,
    )

    aborted = [ids for ids, abort in port.cleanups if abort and request_id in ids]
    assert aborted


@pytest.mark.asyncio
async def test_released_vision_follow_does_not_abort_draining_talker() -> None:
    """After the Stage1 release flag, draining stages 2/3 stay up."""
    session = _ephemeral_session()
    request_id = _bind_inflight_talker(session)
    port = _StagePort()
    run = DuplexRunState(concurrent_turn_requests_released=True)
    channel = _channel(session, port, run)

    await channel._append_via_data_plane(
        {"is_speech": False},
        final=True,
        operation_id="vision-after-release",
        expected_epoch=None,
        input_audio_part=None,
    )

    aborted = [ids for ids, abort in port.cleanups if abort and request_id in ids]
    assert not aborted
    assert session.turn_id == 1
    assert session.request_resources.get((2, request_id)) is not None
    assert session.request_resources.get((3, request_id)) is not None


def _spoken_output(request_id: str):
    completion = SimpleNamespace(
        text=SPOKEN,
        cumulative_text=SPOKEN,
        finished=True,
        token_ids=[11, 22, 33],
    )
    return SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        text=SPOKEN,
        cumulative_text=SPOKEN,
        outputs=[completion],
    )


@pytest.mark.asyncio
async def test_finished_stage1_handoff_submit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sentence TTS on and off: does Talker get submit_initial, or a dropped update?"""
    from tests.engine.test_duplex_orchestrator import _build

    monkeypatch.delenv("VLLM_AURA_SENTENCE_TTS", raising=False)
    orchestrator, clients, _, _ = _build(stages=4, plugin=AuraDuplexPlugin(lambda *args: None))
    orchestrator.async_chunk = True
    calls: list[tuple[str, object]] = []

    def aura2tts(*args, **kwargs):
        del args, kwargs
        return [{"prompt_token_ids": [11, 22, 33], "additional_information": {"text": SPOKEN}}]

    clients[2].custom_process_input_func = aura2tts
    clients[2].process_engine_inputs = aura2tts
    pool = orchestrator.stage_pools[2]
    original_initial = pool.submit_initial
    original_update = pool.submit_update

    async def submit_initial(*args, **kwargs):
        calls.append(("initial", args[2] if len(args) > 2 else kwargs.get("request")))
        return await original_initial(*args, **kwargs)

    async def submit_update(*args, **kwargs):
        calls.append(("update", args[2] if len(args) > 2 else kwargs.get("request")))
        return await original_update(*args, **kwargs)

    pool.submit_initial = submit_initial  # type: ignore[method-assign]
    pool.submit_update = submit_update  # type: ignore[method-assign]

    request_id = "duplex-s.cause.e.0.r.stage0-turn0"
    req_state = DuplexOrchestratorRequestState(
        request_id=request_id,
        final_stage_id=3,
        session_owned=True,
        sampling_params_list=[SamplingParams(max_tokens=8) for _ in range(4)],
        prompt={"prompt_token_ids": [1], "additional_information": {}},
    )
    req_state.streaming.enabled = False
    orchestrator.request_states[request_id] = req_state

    await orchestrator._route_output(1, 0, _spoken_output(request_id), req_state, None)

    assert calls, "Stage1 text never reached a Talker submit"
    kind, request = calls[0]
    assert kind == "initial"
    assert getattr(request, "resumable", None) is False
    assert OmniStreamingUpdate.from_request(request) is None
    prompt_ids = list(getattr(request, "prompt_token_ids", []) or [])
    assert prompt_ids, "submit_initial carried an empty prompt"


def _stage1_chunk(request_id: str, text: str, *, finished: bool):
    completion = SimpleNamespace(
        text=text,
        cumulative_text=text,
        finished=finished,
        token_ids=[11, 22, 33],
    )
    return SimpleNamespace(
        request_id=request_id,
        finished=finished,
        stage_id=1,
        text=text,
        cumulative_text=text,
        outputs=[completion],
    )


@pytest.mark.asyncio
async def test_empty_remainder_close_only_forwards_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """17:30: mid-gen full sentence then empty finish must close Talker once."""
    from tests.engine.test_duplex_orchestrator import _build

    monkeypatch.delenv("VLLM_AURA_SENTENCE_TTS", raising=False)
    orchestrator, clients, _, _ = _build(stages=4, plugin=AuraDuplexPlugin(lambda *args: None))
    orchestrator.async_chunk = True
    close_only_forwards: list[bool] = []

    def aura2tts(source_outputs, prompt=None, **kwargs):
        del source_outputs, kwargs
        info = prompt.get("additional_information") if isinstance(prompt, dict) else {}
        if not isinstance(info, dict):
            info = {}
        close_only = bool(info.get("aura_tts_close_only"))
        if close_only:
            close_only_forwards.append(True)
            return [{"prompt_token_ids": [0], "additional_information": {"text": ""}}]
        return [{"prompt_token_ids": [11, 22, 33], "additional_information": {"text": SPOKEN}}]

    clients[2].custom_process_input_func = aura2tts
    clients[2].process_engine_inputs = aura2tts

    request_id = "duplex-s.cause.e.0.r.stage0-turn0"
    req_state = DuplexOrchestratorRequestState(
        request_id=request_id,
        final_stage_id=3,
        session_owned=True,
        sampling_params_list=[SamplingParams(max_tokens=8) for _ in range(4)],
        prompt={"prompt_token_ids": [1], "additional_information": {}},
    )
    req_state.streaming.enabled = False
    orchestrator.request_states[request_id] = req_state
    runners = getattr(orchestrator.session_manager, "runners", None)
    if runners is not None:
        runners["duplex-cause"] = SimpleNamespace(
            run=DuplexRunState(concurrent_turn_requests_released=False),
            session=_ephemeral_session(),
        )
        orchestrator.session_manager.register_request(request_id, "duplex-cause")

    sentence = "看到了，我正盯着呢。现在我看到一个戴眼镜、穿黑T恤的男生，他正戴着耳机坐在办公桌前。"
    await orchestrator._route_output(1, 0, _stage1_chunk(request_id, sentence, finished=False), req_state, None)
    assert close_only_forwards == []
    await orchestrator._route_output(1, 0, _stage1_chunk(request_id, sentence, finished=True), req_state, None)
    assert len(close_only_forwards) == 1


@pytest.mark.asyncio
async def test_sentence_tts_handoff_does_not_abort_before_stage2_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vision commit inside the Talker submit must open the next Stage0 only.

    Stage2 is still unbound and intercept has not set the release flag yet.
    Sentence TTS has already started handing text to Talker, so that id stays.
    """
    from tests.engine.test_duplex_orchestrator import _build

    monkeypatch.delenv("VLLM_AURA_SENTENCE_TTS", raising=False)
    orchestrator, clients, _, _ = _build(stages=4, plugin=AuraDuplexPlugin(lambda *args: None))
    orchestrator.async_chunk = True

    def aura2tts(*args, **kwargs):
        del args, kwargs
        return [{"prompt_token_ids": [11, 22, 33], "additional_information": {"text": SPOKEN}}]

    clients[2].custom_process_input_func = aura2tts
    clients[2].process_engine_inputs = aura2tts

    session = _ephemeral_session()
    request_id = _bind_stage0(session)
    port = _StagePort()
    run = DuplexRunState(concurrent_turn_requests_released=False)
    channel = _channel(session, port, run)
    orchestrator.session_manager.runners[session.session_id] = SimpleNamespace(run=run, session=session)
    orchestrator.session_manager.register_request(request_id, session.session_id)

    original_forward = orchestrator._forward_to_next_stage

    async def forward_then_commit(*args, **kwargs):
        await channel._append_via_data_plane(
            {"is_speech": False},
            final=True,
            operation_id="vision-during-tts",
            expected_epoch=None,
            input_audio_part=None,
        )
        return await original_forward(*args, **kwargs)

    orchestrator._forward_to_next_stage = forward_then_commit  # type: ignore[method-assign]

    req_state = DuplexOrchestratorRequestState(
        request_id=request_id,
        final_stage_id=3,
        session_owned=True,
        sampling_params_list=[SamplingParams(max_tokens=8) for _ in range(4)],
        prompt={"prompt_token_ids": [1], "additional_information": {}},
    )
    req_state.streaming.enabled = False
    orchestrator.request_states[request_id] = req_state

    result = await orchestrator._route_output(1, 0, _spoken_output(request_id), req_state, None)

    del result
    aborted = [ids for ids, abort in port.cleanups if abort and request_id in ids]
    assert not aborted
    assert session.turn_id == 1
    assert not any(stage_id in (2, 3) for stage_id, _ in session.request_resources)


@pytest.mark.asyncio
async def test_sentence_tts_off_still_submits_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing sentence TTS is not a repair: the legacy path must still hand the sentence over."""
    from tests.engine.test_duplex_orchestrator import _build

    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "0")
    orchestrator, clients, _, _ = _build(stages=4, plugin=AuraDuplexPlugin(lambda *args: None))
    orchestrator.async_chunk = True
    calls: list[str] = []

    def aura2tts(*args, **kwargs):
        del args, kwargs
        return [{"prompt_token_ids": [11, 22, 33], "additional_information": {"text": SPOKEN}}]

    clients[2].custom_process_input_func = aura2tts
    clients[2].process_engine_inputs = aura2tts
    pool = orchestrator.stage_pools[2]
    original_initial = pool.submit_initial

    async def submit_initial(*args, **kwargs):
        calls.append("initial")
        return await original_initial(*args, **kwargs)

    pool.submit_initial = submit_initial  # type: ignore[method-assign]

    request_id = "duplex-s.cause.e.0.r.stage0-turn0"
    req_state = DuplexOrchestratorRequestState(
        request_id=request_id,
        final_stage_id=3,
        session_owned=True,
        sampling_params_list=[SamplingParams(max_tokens=8) for _ in range(4)],
        prompt={"prompt_token_ids": [1], "additional_information": {}},
    )
    req_state.streaming.enabled = False
    orchestrator.request_states[request_id] = req_state
    await orchestrator._route_output(1, 0, _spoken_output(request_id), req_state, None)
    assert calls == ["initial"]


def test_project_output_yields_spoken_stage1_text() -> None:
    plane = AuraDataPlaneSession(lambda *args: None)
    request_id = "duplex-s.cause.e.0.r.stage0-turn0"
    plane.begin_request(request_id)
    events = list(plane.project_output(_spoken_output(request_id)))
    texts = [event.get("text") for event in events if isinstance(event.get("text"), str) and event.get("text")]
    assert texts
    assert SPOKEN in texts[0]
    assert all(event.get("is_listen") is not True for event in events)
