# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The response-judge stage on AURA through the real duplex session paths."""

from __future__ import annotations

import importlib
import weakref
from types import SimpleNamespace

import pytest

from tests.engine.test_duplex_orchestrator import (
    SESSION_ID,
    _build,
    _close,
    _encode_audio,
    _open,
    _settle,
    _submit,
)
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, resolve_deploy_yaml
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.contracts import DuplexOutputAction, duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.session.model_channel import ModelChannel
from vllm_omni.model_executor.models.aura_omni.duplex.plugin import (
    AURA_SILENT_TOKEN_ID,
    AuraDuplexPlugin,
    AuraJudgedDuplexPlugin,
)
from vllm_omni.model_executor.models.aura_omni.duplex.stages import AURA_JUDGED_STAGE_LAYOUT, AURA_STAGE_LAYOUT
from vllm_omni.model_executor.models.aura_omni.pipeline import AURA_OMNI_JUDGED_PIPELINE, AURA_OMNI_PIPELINE
from vllm_omni.model_executor.models.registry import OmniModelRegistry
from vllm_omni.model_executor.stage_input_processors import response_judge as rj
from vllm_omni.model_executor.stage_input_processors.aura_omni import asr2aura, asr2judge, judge2aura

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_JUDGE = AURA_JUDGED_STAGE_LAYOUT.judge


def _resolve(path: str):
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def _judged_orchestrator(**kwargs):
    orchestrator, clients, rpc_q, output_q = _build(stages=5, plugin=AuraJudgedDuplexPlugin(_encode_audio))
    orchestrator.stage_pools[_JUDGE].stage_vllm_config.model_config.model_stage = rj.RESPONSE_JUDGE_STAGE
    return orchestrator, clients, rpc_q, output_q


async def _pending_judge(orchestrator):
    request_id = next(iter(orchestrator.request_states))
    assert duplex_resource_request_belongs_to_session(request_id, SESSION_ID)
    state = orchestrator.request_states[request_id]
    asr = SimpleNamespace(request_id=request_id, outputs=[SimpleNamespace(text="嗯嗯")])
    rj._remember(request_id, rj._PendingTurn(rj.JudgeSpec("chat_yes_no", {}), "嗯嗯", asr), state.streaming)
    request = SimpleNamespace(request_id=request_id, prompt_token_ids=[1], resumable=False)
    replica_id = await orchestrator.stage_pools[_JUDGE].submit_initial(request_id, state, request)
    orchestrator._on_stage_submitted(_JUDGE, request_id, replica_id, state)
    return request_id


async def _pending_at(orchestrator, stage_id):
    request_id = next(iter(orchestrator.request_states))
    state = orchestrator.request_states[request_id]
    request = SimpleNamespace(request_id=request_id, prompt_token_ids=[1], resumable=False)
    replica_id = await orchestrator.stage_pools[stage_id].submit_initial(request_id, state, request)
    orchestrator._on_stage_submitted(stage_id, request_id, replica_id, state)
    return request_id


def _judge_output(request_id: str, text: str):
    return SimpleNamespace(request_id=request_id, finished=True, outputs=[SimpleNamespace(text=text, token_ids=[1])])


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_response", [True, False])
async def test_judge_no_ends_the_turn_through_the_native_listen_path(auto_response):
    orchestrator, clients, rpc_q, _ = _judged_orchestrator()
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    try:
        assert (
            await _open(orchestrator, rpc_q, extra_body={"auto_response": auto_response}, output_buffer=output_buffer)
        ).ok
        request_id = await _pending_judge(orchestrator)
        pending = weakref.ref(rj._owned_turn(request_id, orchestrator.request_states[request_id].streaming))
        session = orchestrator.session_manager.get(SESSION_ID)
        session.bind_request(request_id)
        session.bind_response_turn(session.turn_id)
        response_id = session.begin_response(turn_id=session.turn_id)
        consumed = await orchestrator._intercept_stage_output(
            _JUDGE, 0, _judge_output(request_id, "NO"), orchestrator.request_states[request_id], None, None
        )
        assert consumed
        await _settle(orchestrator)
        assert request_id not in orchestrator.request_states
        assert pending() is None  # released with the request
        assert clients[2].add_request_calls == []
        assert clients[3].add_request_calls == []
        events = [await output_buffer.get() for _ in range(output_buffer.pending_events)]
        types = [event.type for event in events]
        assert "error" not in types
        assert "response.listen" in types
        listen = next(event for event in events if event.type == "response.listen").to_realtime()
        # Clients can tell a judge rejection from AURA's own <|silent|>.
        assert listen["response"]["metadata"]["vllm_omni"]["listen_source"] == "response_judge"
        done = [event for event in events if event.type == "response.done"]
        assert [event.response_id for event in done] == [response_id]
        assert session.active_response_id is None
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_aura_silence_reports_its_own_listen_source():
    """Without a judge, AURA's own <|silent|> reaches the client as "aura_silent", not "response_judge"."""
    orchestrator, clients, rpc_q, _ = _build(stages=4, plugin=AuraDuplexPlugin(_encode_audio))
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    try:
        assert (await _open(orchestrator, rpc_q, output_buffer=output_buffer)).ok
        aura = AURA_STAGE_LAYOUT.aura
        request_id = await _pending_at(orchestrator, aura)
        silent = SimpleNamespace(
            request_id=request_id,
            finished=True,
            outputs=[SimpleNamespace(text="<|silent|>", token_ids=[AURA_SILENT_TOKEN_ID])],
        )
        assert await orchestrator._intercept_stage_output(
            aura, 0, silent, orchestrator.request_states[request_id], None, None
        )
        await _settle(orchestrator)
        events = [await output_buffer.get() for _ in range(output_buffer.pending_events)]
        listen = next(event for event in events if event.type == "response.listen").to_realtime()
        assert listen["response"]["metadata"]["vllm_omni"]["listen_source"] == "aura_silent"
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.parametrize(
    ("model_result", "expected"),
    [
        ({"listen_source": "response_judge"}, "response_judge"),
        ({}, None),
        ({"listen_source": ""}, None),
        ({"listen_source": 1}, None),
    ],
)
def test_runtime_metadata_forwards_only_a_named_listen_source(model_result, expected):
    payload: dict[str, object] = {}
    ModelChannel._attach_runtime_metadata(payload, {"model_turn_id": 1, **model_result})
    assert payload["vllm_omni"].get("listen_source") == expected


@pytest.mark.asyncio
async def test_judge_yes_is_not_consumed_by_the_session():
    orchestrator, clients, rpc_q, _ = _judged_orchestrator()
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        request_id = await _pending_judge(orchestrator)
        consumed = await orchestrator._intercept_stage_output(
            _JUDGE, 0, _judge_output(request_id, "YES"), orchestrator.request_states[request_id], None, None
        )
        assert not consumed
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_no_from_a_stage_that_is_not_a_judge_is_ignored():
    orchestrator, clients, rpc_q, _ = _build(stages=5, plugin=AuraJudgedDuplexPlugin(_encode_audio))
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        request_id = await _pending_judge(orchestrator)
        consumed = await orchestrator._intercept_stage_output(
            _JUDGE, 0, _judge_output(request_id, "NO"), orchestrator.request_states[request_id], None, None
        )
        assert not consumed
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["cancel", "close"])
async def test_native_terminal_paths_release_the_pending_turn(terminal):
    orchestrator, clients, rpc_q, _ = _judged_orchestrator()
    try:
        assert (await _open(orchestrator, rpc_q)).ok
        request_id = await _pending_judge(orchestrator)
        pending = weakref.ref(rj._owned_turn(request_id, orchestrator.request_states[request_id].streaming))
        assert (
            rj.judge_rejects(_judge_output(request_id, "NO"), orchestrator.request_states[request_id].streaming) is True
        )
        if terminal == "cancel":
            await _submit(orchestrator, commands.BargeIn())
        else:
            assert (await _close(orchestrator, rpc_q)).ok
        await _settle(orchestrator)
        assert request_id not in orchestrator.request_states
        assert pending() is None
        assert clients[2].add_request_calls == []
    finally:
        await orchestrator.session_manager.shutdown()


def test_default_plugin_decision_for_a_rejected_turn_is_a_model_listen():
    decision = AuraJudgedDuplexPlugin(_encode_audio).response_judge_decision()
    assert decision.action == DuplexOutputAction.DIRECT_RESPONSE
    assert decision.metadata["model_listen"] is True
    assert decision.metadata["listen_source"] == "response_judge"


def test_default_aura_keeps_its_stage_ids():
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.stages == AURA_STAGE_LAYOUT
    assert plugin.draining_stage_ids(stage_count=4) == frozenset({2, 3})
    assert plugin.project_intermediate_output(stage_id=1, output=None, context=None) is True


def test_judged_aura_shifts_the_stages_after_the_judge():
    plugin = AuraJudgedDuplexPlugin(_encode_audio)
    assert plugin.draining_stage_ids(stage_count=5) == frozenset({3, 4})
    assert plugin.project_intermediate_output(stage_id=2, output=None, context=None) is True
    assert plugin.project_intermediate_output(stage_id=1, output=None, context=None) is False


def test_judged_pipeline_routes_asr_through_the_judge_before_aura():
    pipeline = resolve_pipeline_config("aura_omni_judged")
    assert pipeline is AURA_OMNI_JUDGED_PIPELINE
    assert [s.model_stage for s in pipeline.stages] == ["asr", "response_judge", "aura", "qwen3_tts", "code2wav"]
    assert [s.input_sources for s in pipeline.stages] == [(), (0,), (1,), (2,), (3,)]
    assert _resolve(pipeline.stages[1].custom_process_input_func) is asr2judge
    assert _resolve(pipeline.stages[2].custom_process_input_func) is judge2aura
    assert resolve_pipeline_config("aura_omni") is AURA_OMNI_PIPELINE
    assert _resolve(AURA_OMNI_PIPELINE.stages[1].custom_process_input_func) is asr2aura


@pytest.mark.parametrize("arch", ["ResponseJudgeQwen3ForCausalLM", "LayaDecisionModel", "ClmDecisionModel"])
def test_judge_architectures_are_registered(arch):
    assert OmniModelRegistry._try_load_model_cls(arch) is not None


def test_judged_deploy_puts_a_one_token_judge_on_stage_1():
    pipeline = resolve_pipeline_config("aura_omni_judged")
    path = get_deploy_config_path(pipeline.default_deploy_config_name)
    assert load_deploy_config(path).session_mode == "duplex"
    stages, _ = StageConfigFactory._create_legacy_from_registry(pipeline, cli_overrides={}, deploy_config_path=path)
    judge = stages[1]
    assert judge.yaml_engine_args["model_arch"] == "ResponseJudgeQwen3ForCausalLM"
    assert judge.yaml_engine_args["model"] == "Qwen/Qwen3-1.7B"
    assert judge.yaml_engine_args["hf_overrides"]["response_judge"]["format"] == "chat_yes_no"
    assert judge.yaml_extras["default_sampling_params"]["max_tokens"] == 1
    assert stages[3].yaml_extras["output_connectors"] == {"to_stage_4": "connector_of_shared_memory"}


def test_laya_overlay_only_changes_the_judge_stage():
    base = resolve_deploy_yaml(get_deploy_config_path("aura_omni_judged.yaml"))
    laya = resolve_deploy_yaml(get_deploy_config_path("aura_omni_judged_laya.yaml"))
    assert [s for s in laya["stages"] if s["stage_id"] != 1] == [s for s in base["stages"] if s["stage_id"] != 1]
    judge = next(s for s in laya["stages"] if s["stage_id"] == 1)
    assert judge["model_arch"] == "LayaDecisionModel"
    assert judge["runner"] == "pooling"
    assert judge["hf_overrides"]["response_judge"]["format"] == "laya"
    # Capture sizes reach past the measured judge prompts (up to about 170 tokens), not only decode-sized batches.
    assert max(judge["compilation_config"]["cudagraph_capture_sizes"]) >= 192
    options = judge["hf_overrides"]["response_judge"]
    assert options["reply_option"] in options["options"]


def test_clm_overlay_only_changes_the_judge_stage():
    base = resolve_deploy_yaml(get_deploy_config_path("aura_omni_judged.yaml"))
    clm = resolve_deploy_yaml(get_deploy_config_path("aura_omni_judged_clm.yaml"))
    assert [s for s in clm["stages"] if s["stage_id"] != 1] == [s for s in base["stages"] if s["stage_id"] != 1]
    judge = next(s for s in clm["stages"] if s["stage_id"] == 1)
    assert judge["model_arch"] == "ClmDecisionModel"
    assert judge["runner"] == "pooling"
    assert judge["default_pooling_params"] == {"task": "classify"}
    # Capture sizes reach past the measured CLM prompts (about 36 tokens).
    assert max(judge["compilation_config"]["cudagraph_capture_sizes"]) >= 64
    # The prepared model directory's config.json carries format=clm; the base
    # file's chat_yes_no options must not be inherited on top of it.
    assert judge["hf_overrides"] == {}
    pipeline = resolve_pipeline_config("aura_omni_judged")
    stages, _ = StageConfigFactory._create_legacy_from_registry(
        pipeline, cli_overrides={}, deploy_config_path=get_deploy_config_path("aura_omni_judged_clm.yaml")
    )
    assert "response_judge" not in (stages[1].yaml_engine_args.get("hf_overrides") or {})


def _pool_with_decoder(name):
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: f"{name}:{len(ids)}")
    return SimpleNamespace(output_processor=SimpleNamespace(tokenizer=tokenizer))


def test_sentence_tts_decodes_aura_tokens_with_the_aura_stage_tokenizer():
    from vllm_omni.model_executor.models.aura_omni.duplex import sentence_tts

    orchestrator = SimpleNamespace(stage_pools=[None, _pool_with_decoder("judge"), _pool_with_decoder("aura")])
    # Mid-generation AURA chunk: token ids only, no cumulative text yet.
    output = SimpleNamespace(outputs=[SimpleNamespace(cumulative_text="", cumulative_token_ids=[5, 6, 7], text="")])
    text = sentence_tts.stage1_tts_text(orchestrator, output, aura_stage_id=AURA_JUDGED_STAGE_LAYOUT.aura)
    assert text == "aura:3"


def test_partial_sentence_planning_passes_the_aura_stage_id(monkeypatch):
    from vllm_omni.model_executor.models.aura_omni.duplex import sentence_tts

    seen = []

    def fake_text(orchestrator, output, *, cache=None, aura_stage_id=1):
        seen.append(aura_stage_id)
        return ""

    monkeypatch.setattr(sentence_tts, "stage1_tts_text", fake_text)
    aura2tts = SimpleNamespace(__name__="aura2tts")
    orchestrator = SimpleNamespace(
        stage_pools=[
            None,
            None,
            None,
            SimpleNamespace(stage_client=SimpleNamespace(custom_process_input_func=aura2tts)),
        ],
        _stage_receives_async_chunks=lambda stage_id: False,
    )
    req_state = SimpleNamespace(session_owned=True, final_stage_id=4, streaming=SimpleNamespace(bridge_states={}))
    plugin = AuraJudgedDuplexPlugin(_encode_audio)
    output = SimpleNamespace(finished=False, outputs=[SimpleNamespace(text="")])
    plugin.plan_partial_stage_output(orchestrator, AURA_JUDGED_STAGE_LAYOUT.aura, 0, output, req_state)
    assert seen == [AURA_JUDGED_STAGE_LAYOUT.aura]
