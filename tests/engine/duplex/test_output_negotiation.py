# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deployment output negotiation and text response lifecycle (RFC #10143)."""

import copy
from dataclasses import replace

import pytest

from tests.engine.duplex.test_session_manager import FakePlugin, FakeStagePort, Harness
from tests.engine.duplex.test_session_runner import _fake_encode_audio, close_harness, open_harness
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import (
    DuplexCapabilities,
    DuplexConfigError,
    DuplexSessionConfig,
    ResponseCreateOptions,
)
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin as minicpm_plugin
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TextPlugin(FakePlugin):
    def configure_sampling_params(self, *, runtime_config, defaults):
        return defaults

    def history_calibrator(self, runtime_config):
        raise AssertionError("Text-only deployment must not start audio calibration")


def _text_plugin():
    plugin = _TextPlugin()
    plugin.configure_pipeline(replace(MINICPMO_4_5_PIPELINE, stages=MINICPMO_4_5_PIPELINE.stages[:1]))
    return plugin


def _caps():
    return _text_plugin().deployment_capabilities(max_sessions=2)


def test_effective_capabilities_follow_topology_and_preserve_inputs():
    plugin = _text_plugin()
    caps = plugin.deployment_capabilities(max_sessions=2)
    assert caps.supported_output_modalities == {"text"}
    assert caps.default_output_modalities == ("text",)
    assert not caps.supports_playback_ack
    assert not caps.supports_audio_truncate
    assert not caps.supports_stage_connector_handoff
    assert caps.stage_handoff_transport is None
    assert caps.required_input_modalities == plugin.capabilities(max_sessions=2).required_input_modalities
    assert caps.as_dict()["default_output_modalities"] == ["text"]
    plugin.configure_pipeline(MINICPMO_4_5_PIPELINE)
    assert plugin.deployment_capabilities(max_sessions=2) == plugin.capabilities(max_sessions=2)


@pytest.mark.asyncio
async def test_real_minicpm_text_topology_defaults_without_reference_audio(monkeypatch):
    async def reject_default_audio(model_config):
        raise AssertionError("Text deployment must not load the default reference audio")

    monkeypatch.setattr(minicpm_plugin, "resolve_default_ref_audio", reject_default_audio)
    monkeypatch.setattr(minicpm_plugin, "_load_tokenizer", lambda model_config: None)
    plugin = minicpm_plugin.MiniCPMO45DuplexPlugin(_fake_encode_audio)
    plugin.configure_pipeline(replace(MINICPMO_4_5_PIPELINE, stages=MINICPMO_4_5_PIPELINE.stages[:1]))
    assert plugin.deployment_capabilities(max_sessions=1).default_output_modalities == ("text",)
    runtime = await plugin.prepare_runtime_config(DuplexSessionConfig(), model_config=None)
    assert "ref_audio_data" not in runtime


@pytest.mark.parametrize("factory", [DuplexSessionConfig.from_event, DuplexSessionConfig.from_realtime])
def test_omission_survives_normalization_until_deployment_is_known(factory):
    config = factory({})
    assert config.modalities is None
    assert config.normalized().modalities is None
    assert config.normalized().with_output_capabilities(_caps()).modalities == ["text"]
    assert config.with_output_capabilities(DuplexCapabilities()).modalities == ["text", "audio"]


@pytest.mark.parametrize("field", ["modalities", "output_modalities"])
@pytest.mark.parametrize("value", [[], None, "text", ["image"], [1], ["text", "text"], ["Audio", "audio"]])
@pytest.mark.parametrize(
    "factory", [DuplexSessionConfig.from_event, DuplexSessionConfig.from_realtime, ResponseCreateOptions.from_realtime]
)
def test_invalid_explicit_modalities_are_never_defaults(field, value, factory):
    with pytest.raises(DuplexConfigError):
        factory({field: value})


@pytest.mark.parametrize(
    "factory", [DuplexSessionConfig.from_event, DuplexSessionConfig.from_realtime, ResponseCreateOptions.from_realtime]
)
def test_conflicting_aliases_are_rejected(factory):
    with pytest.raises(DuplexConfigError):
        factory({"modalities": ["text"], "output_modalities": ["audio"]})


@pytest.mark.parametrize("field", ["modalities", "output_modalities"])
@pytest.mark.parametrize("requested", [["audio"], ["text", "audio"]])
@pytest.mark.asyncio
async def test_open_rejects_audio_before_allocating_stage_resources(field, requested):
    async with Harness.create(plugin=_text_plugin(), stage_port=FakeStagePort(stage_count=1)) as h:
        config = DuplexSessionConfig.from_realtime({field: requested})
        before = copy.deepcopy(config.as_dict())
        result = await h.open("rejected", config)
        assert not result.ok and result.error_code == "unsupported_output_modality"
        assert not h.stage_port.ensure_calls
        assert not h.stage_port.submit_calls
        assert not h.manager.runners
        assert config.as_dict() == before


@pytest.mark.asyncio
async def test_open_defaults_to_text_and_close_cleans_only_deployed_stage():
    async with Harness.create(plugin=_text_plugin(), stage_port=FakeStagePort(stage_count=1)) as h:
        result = await h.open("text")
        assert result.ok
        session = h.manager.runners["text"].session
        assert session.config.modalities == ["text"]
        assert session.num_stages == 1
        assert [call.stage_id for call in h.stage_port.ensure_calls] == [0]
        assert (await h.close("text")).ok
        assert not session.request_resources


@pytest.mark.parametrize("policy", ["ack_only", "commit_all_on_done"])
@pytest.mark.parametrize("complete_audio_metadata", [False, True])
def test_text_history_does_not_wait_for_playback(policy, complete_audio_metadata):
    session = DuplexEngineSession(
        session_id="text", config=DuplexSessionConfig(playback_commit_policy=policy), capabilities=_caps()
    )
    response = session.begin_response()
    session.append_assistant_text(" emitted prefix ")
    session.mark_audio_sent(None, text_requires_complete_audio=complete_audio_metadata)
    message = session.end_response(playback_commit_policy="ack_only")
    session.register_history_item(f"item_{response}", message)
    session.finalize_response_playback(response)
    assert session.history == ({"role": "assistant", "content": " emitted prefix "},)
    assert not session.pending_history_item_ids
    assert not session._playback.by_response
    assert not session._conversation.assistant_response_snapshots
    before = copy.deepcopy(session.history)
    events = apply_playback_ack(session, {"response_id": response, "played_ms": 500})
    assert events[0].type == "error" and events[0].code == "unsupported_operation"
    assert session.history == before


@pytest.mark.parametrize("text", ["", " ", "\n", "\t \n"])
@pytest.mark.parametrize("policy", ["ack_only", "commit_all_on_done"])
def test_whitespace_only_text_does_not_create_history_or_snapshots(text, policy):
    session = DuplexEngineSession(
        session_id="blank", config=DuplexSessionConfig(playback_commit_policy=policy), capabilities=_caps()
    )
    response = session.begin_response()
    session.append_assistant_text(text)
    assert session.end_response() is None
    assert session.history == ()
    assert not session.pending_history_item_ids
    assert not session._conversation.assistant_response_snapshots
    assert not session._conversation.history_item_placeholders
    session.finalize_response_playback(response)
    assert not session._playback.by_response


@pytest.mark.parametrize("text_override", [False, True])
def test_speech_deployment_keeps_playback_policy_for_text_response_override(text_override):
    session = DuplexEngineSession(session_id="speech", config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    if text_override:
        session.reserve_response_options(ResponseCreateOptions(modalities=("text",)))
    response = session.begin_response()
    session.append_assistant_text(" emitted text ")
    session.mark_audio_sent(None, text_requires_complete_audio=True)
    # A text request on a speech deployment does not establish TTS bypass.
    assert session.end_response() is None
    assert session.history == ()
    assert session._conversation.assistant_response_snapshots[response][0]["content"] == "emitted text"


@pytest.mark.parametrize("field", ["modalities", "output_modalities"])
@pytest.mark.parametrize(
    "factory", [DuplexSessionConfig.from_event, DuplexSessionConfig.from_realtime, ResponseCreateOptions.from_realtime]
)
def test_mixed_case_modalities_are_normalized(factory, field):
    config = factory({field: ["Audio", "Text"]})
    assert config.modalities is not None
    assert list(config.modalities) == ["audio", "text"]
    config = factory({"modalities": ["Text"], "output_modalities": ["TEXT"]})
    assert config.modalities is not None
    assert list(config.modalities) == ["text"]


@pytest.mark.asyncio
async def test_mixed_case_modalities_normalized_on_live_update_and_response():
    h = await open_harness(plugin=_text_plugin(), stage_count=1)
    try:
        events = await h.run(commands.UpdateSession(patch={"output_modalities": ["Text"]}))
        assert not any(event.type == "error" for event in events)
        assert h.session.config.modalities == ["text"]
        events = await h.run(commands.CreateResponse(options={"modalities": ["Audio"]}))
        assert len(events) == 1 and events[0].type == "error"
        assert events[0].code == "unsupported_output_modality"
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.asyncio
async def test_audio_capable_deployment_can_disable_truncation_without_history_changes(native):
    h = await open_harness()
    try:
        assert h.session.capabilities.emits_audio
        h.session.capabilities = replace(h.session.capabilities, supports_audio_truncate=False)
        response = h.session.begin_response()
        h.session.append_assistant_text("keep this reply")
        h.session.mark_audio_sent(100, audio_complete=True)
        before = copy.deepcopy((h.session.history, h.session._playback, h.session._conversation))
        command = (
            commands.SignalTurn(
                event="conversation.item.truncate",
                signal_payload={"item_id": f"item_{response}", "audio_end_ms": 0},
                event_id="unsupported-cut",
            )
            if native
            else commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=0, event_id="unsupported-cut")
        )
        events = await h.run(command)
        assert len(events) == 1 and events[0].type == "error"
        assert events[0].code == "unsupported_operation"
        assert events[0].related_event_id == "unsupported-cut"
        assert (h.session.history, h.session._playback, h.session._conversation) == before
    finally:
        await close_harness(h)


@pytest.mark.parametrize("field", ["modalities", "output_modalities"])
def test_response_override_rejected_without_reserving_options(field):
    session = DuplexEngineSession(session_id="text", config=DuplexSessionConfig(), capabilities=_caps())
    before = session.config.as_dict()
    with pytest.raises(DuplexConfigError, match="audio"):
        session.reserve_response_options(ResponseCreateOptions.from_realtime({field: ["audio"]}))
    assert session.config.as_dict() == before
    assert session._response.pending_options is None
    assert session.active_response_id is None


@pytest.mark.parametrize("field", ["modalities", "output_modalities"])
@pytest.mark.asyncio
async def test_response_create_rejects_audio_before_request_submission(field):
    h = await open_harness(plugin=_text_plugin(), stage_count=1)
    try:
        ensured = list(h.port.ensured)
        events = await h.run(commands.CreateResponse(options={field: ["audio"]}, event_id="reject-audio"))
        assert len(events) == 1 and events[0].type == "error"
        assert events[0].code == "unsupported_output_modality"
        assert events[0].related_event_id == "reject-audio"
        assert "audio" in events[0].message
        assert h.port.ensured == ensured
        assert not h.port.submissions
        assert h.session.active_response_id is None
        assert h.session._response.pending_options is None
    finally:
        await close_harness(h)


@pytest.mark.parametrize("patch", [{"voice": "alloy"}, {"speed": 1.2}])
def test_synthesis_parameters_rejected_for_text_deployment(patch):
    with pytest.raises(DuplexConfigError) as exc:
        DuplexSessionConfig.from_realtime(patch).with_output_capabilities(_caps())
    assert exc.value.code == "unsupported_parameter"
    with pytest.raises(DuplexConfigError) as exc:
        ResponseCreateOptions.from_realtime(patch, capabilities=_caps())
    assert exc.value.code == "unsupported_parameter"


@pytest.mark.parametrize(
    "patch",
    [
        {"modalities": ["audio"]},
        {"output_modalities": ["audio"]},
        {"modalities": []},
        {"modalities": ["text"], "output_modalities": ["audio"]},
        {"voice": "alloy"},
        {"audio": {"output": {"speed": 1.5}}},
    ],
)
@pytest.mark.asyncio
async def test_session_update_is_atomic(patch):
    h = await open_harness(plugin=_text_plugin(), stage_count=1)
    try:
        before = (copy.deepcopy(h.session.config.as_dict()), dict(h.session.runtime_config), list(h.port.ensured))
        events = await h.run(commands.UpdateSession(patch={"instructions": "must not apply", **patch}))
        assert len(events) == 1 and events[0].type == "error"
        assert (h.session.config.as_dict(), h.session.runtime_config, h.port.ensured) == before
    finally:
        await close_harness(h)


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("native_truncate", [False, True])
@pytest.mark.asyncio
async def test_text_runner_completion_cancel_and_late_epoch(cancel, native_truncate):
    h = await open_harness(plugin=_text_plugin(), stage_count=1)
    try:
        h.session.replace_config(replace(h.session.config, playback_commit_policy="ack_only"))
        old_epoch = h.session.epoch
        await h.runner.model._send_one_model_output_event({"text": "emitted prefix", "model_turn_id": 0})
        first = await h.settle()
        response = h.session.active_response_id
        assert response is not None
        assert any(e.type == "response.output_text.delta" for e in first)
        if cancel:
            terminal = await h.run(commands.CancelResponse())
            await h.runner.model._send_one_model_output_event(
                {"text": "late tail", "model_turn_id": 0, "end_of_turn": True}, expected_epoch=old_epoch
            )
            assert not await h.settle()
        else:
            await h.runner.model._send_one_model_output_event({"model_turn_id": 0, "end_of_turn": True})
            terminal = await h.settle()
        assert len([e for e in terminal if e.type == "response.done"]) == 1
        assert h.session.history == ({"role": "assistant", "content": "emitted prefix"},)
        assert h.session.active_response_id is None
        assert not h.session.pending_history_item_ids
        assert not h.session._playback.by_response
        assert not any(e.type.startswith("response.output_audio") for e in first + terminal)
        before = copy.deepcopy(h.session.history)
        truncate = (
            commands.SignalTurn(
                event="conversation.item.truncate", signal_payload={"item_id": f"item_{response}", "audio_end_ms": 0}
            )
            if native_truncate
            else commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=0)
        )
        events = await h.run(truncate)
        assert events[0].type == "error" and events[0].code == "unsupported_operation"
        assert h.session.history == before
    finally:
        await close_harness(h)


def test_text_history_keeps_position_while_input_continues():
    session = DuplexEngineSession(session_id="text", config=DuplexSessionConfig(), capabilities=_caps())
    first = session.commit_audio_input(transcript="first")
    response = session.begin_response()
    session.append_assistant_text("reply")
    second = session.commit_audio_input(transcript="next")
    session.register_history_item(f"item_{response}", session.end_response(playback_commit_policy="ack_only"))
    session.finalize_response_playback(response)
    assert session.history == (first.message, {"role": "assistant", "content": "reply"}, second.message)
    assert not session._conversation.history_item_placeholders


@pytest.mark.asyncio
async def test_text_sessions_resume_without_generation_and_expire_independently():
    async with Harness.create(plugin=_text_plugin(), stage_port=FakeStagePort(stage_count=1), idle_ttl_s=2.0) as h:
        assert (await h.open("first")).ok
        assert (await h.open("second")).ok
        first = h.session("first")
        second = h.session("second")
        response = first.begin_response()
        first.append_assistant_text("only first")
        first.register_history_item(f"item_{response}", first.end_response())
        first.finalize_response_playback(response)
        assert second.history == ()
        ensured = list(h.stage_port.ensure_calls)
        assert (await h.resume("first", expected_lease_generation=0)).ok
        assert first.history == ({"role": "assistant", "content": "only first"},)
        assert h.stage_port.ensure_calls == ensured
        assert not h.stage_port.submit_calls
        h.clock.advance(2.1)
        assert await h.manager.reap_expired() == 2
        assert not first.request_resources and not second.request_resources
        assert not h.manager.runners
