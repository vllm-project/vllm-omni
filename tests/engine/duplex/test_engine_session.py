# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-state semantics of the engine-resident duplex session (``DuplexEngineSession``)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex.config import (
    DuplexCapabilities,
    DuplexConfigError,
    DuplexOverlapPolicy,
    DuplexSessionConfig,
    DuplexTurnEventType,
    DuplexTurnState,
    ResponseCreateOptions,
)
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.events import TurnEvent
from vllm_omni.engine.duplex.session import DuplexEngineSession, DuplexFenceMismatchError
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import (
    minicpmo45_native_capabilities,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _session(session_id: str = "duplex-test", config: DuplexSessionConfig | None = None) -> DuplexEngineSession:
    return DuplexEngineSession(session_id=session_id, config=config or DuplexSessionConfig(model="test-model"))


def test_commit_audio_input_does_not_advance_model_turn_identity():
    session = _session()

    first = session.commit_audio_input(transcript="first chunk")
    second = session.commit_audio_input(transcript="second chunk")

    assert session.input_commit_seq == 2
    assert first.input_commit_seq == 1
    assert second.input_commit_seq == 2
    assert first.turn_id == second.turn_id == session.turn_id == 0
    assert first.message["transcript"] == "first chunk"
    assert first.message["content"] == [
        {"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}, "transcript": "first chunk"}
    ]
    assert len(session.history) == 2
    assert session.turn_state == DuplexTurnState.USER_COMMITTED


def test_commit_audio_input_accepts_explicit_turn_id():
    session = _session()

    committed = session.commit_audio_input(turn_id=4)

    assert committed.turn_id == 4
    assert session.turn_id == 0
    assert "transcript" not in committed.message


# ---- response options ----


def test_response_options_apply_to_one_response_without_mutating_session_defaults():
    session = _session(config=DuplexSessionConfig(instructions="base", voice="base-voice", max_tokens=64))
    session.reserve_response_options(
        ResponseCreateOptions(instructions="one response", voice="override-voice", max_tokens=8)
    )

    assert session.config.instructions == "base"
    assert session.config.voice == "base-voice"
    assert session.response_config is session.config
    session.begin_response()
    assert session.config.instructions == "base"
    assert session.config.voice == "base-voice"
    assert session.response_config.instructions == "one response"
    assert session.response_config.voice == "override-voice"
    assert session.response_config.max_tokens == 8

    session.end_response()
    assert session.response_config is session.config
    assert session.config.instructions == "base"
    assert session.config.voice == "base-voice"
    assert session.config.max_tokens == 64


def test_response_options_cannot_overwrite_an_unconsumed_reservation():
    session = _session(config=DuplexSessionConfig(instructions="base"))
    session.reserve_response_options(ResponseCreateOptions(instructions="first"))

    with pytest.raises(RuntimeError, match="already reserved"):
        session.reserve_response_options(ResponseCreateOptions(instructions="second"))

    session.begin_response()
    assert session.response_config.instructions == "first"


def test_response_options_cannot_be_reserved_while_response_is_active():
    session = _session(config=DuplexSessionConfig(instructions="base"))
    session.begin_response()

    with pytest.raises(RuntimeError, match="active"):
        session.reserve_response_options(ResponseCreateOptions(instructions="too late"))


def test_discarded_response_options_do_not_apply():
    session = _session(config=DuplexSessionConfig(instructions="base"))
    session.reserve_response_options(ResponseCreateOptions(instructions="dropped"))
    session.discard_response_options()

    session.begin_response()

    assert session.response_config.instructions == "base"


# ---- response / overlap identity ----


def test_session_owns_response_and_overlap_identity():
    session = _session()

    response_id = session.begin_response()
    assert session.turn_state == DuplexTurnState.ASSISTANT_GENERATING
    session.accumulate_overlap_speech(320)
    session.accumulate_overlap_speech(180)
    session.end_response()

    assert session.active_response_id is None
    assert session.last_response_id == response_id
    assert session.overlap_speech_ms == 500
    assert session.reset_overlap_speech() == 500
    assert session.overlap_speech_ms == 0
    assert session.turn_state == DuplexTurnState.IDLE


def test_session_composes_single_owner_ledgers_with_immutable_views():
    session = _session()

    session.bind_request("req-1")
    response_id = session.begin_response(turn_id=3)
    session.mark_audio_sent(duration_ms=240)

    assert session.active_request_id == "req-1"
    assert session.active_response_id == response_id
    assert session.active_response_turn_id == 3
    assert session.playback.sent_ms == 240
    assert session.turn_state == DuplexTurnState.ASSISTANT_PLAYING
    with pytest.raises(FrozenInstanceError):
        session.playback.sent_ms = 480  # type: ignore[misc]
    assert session.playback.sent_ms == 240


def test_barge_in_advances_epoch_and_drops_uncommitted_assistant_text():
    session = _session()
    response_id = session.begin_response()
    session.bind_request("chatcmpl-duplex-test")
    session.append_assistant_text("unplayed answer")

    new_epoch = session.barge_in()

    assert response_id is not None
    assert new_epoch == 1
    assert session.epoch == 1
    assert session.accepted_fence == DuplexFence(session.session_id, epoch=1, turn_id=0)
    assert session.active_request_id is None
    assert session.active_response_id is None
    assert session.assistant_text_buffer == ()
    assert session.history == ()
    assert session.turn_state == DuplexTurnState.BARGE_IN


# ---- playback ledger ----


def test_playback_ack_tracks_committed_cursor_separately():
    session = _session()
    session.mark_audio_sent(duration_ms=10_000)

    session.acknowledge_playback(played_ms=2_000)

    assert session.playback.generated_ms == 10_000
    assert session.playback.sent_ms == 10_000
    assert session.playback.played_ms == 2_000
    assert session.playback.committed_ms == 2_000
    assert session.playback.as_dict() == {
        "generated_ms": 10_000,
        "sent_ms": 10_000,
        "played_ms": 2_000,
        "committed_ms": 2_000,
    }


def test_playback_ack_is_scoped_per_response():
    session = _session()
    first = session.begin_response()
    session.mark_audio_sent(duration_ms=3_000)
    session.end_response()
    second = session.begin_response()
    session.mark_audio_sent(duration_ms=1_000)

    session.acknowledge_playback(played_ms=2_500, response_id=first)

    assert session.playback_for_response(first).played_ms == 2_500
    assert session.playback_for_response(second).played_ms == 0
    assert session.playback.played_ms == 0


def test_history_commit_uses_audio_text_alignment_marks():
    session = _session()
    session.begin_response()

    session.append_assistant_text("hello ")
    session.mark_audio_sent(duration_ms=1_000, text_chars=6)
    session.append_assistant_text("world")
    session.mark_audio_sent(duration_ms=2_000, text_chars=11)
    assert [(mark.text_chars, mark.audio_end_ms) for mark in session.assistant_audio_text_marks] == [
        (6, 1_000),
        (11, 2_000),
    ]
    session.acknowledge_playback(played_ms=1_200, committed_ms=1_200)

    committed = session.end_response(commit_text=True)

    assert committed == {"role": "assistant", "content": "hello w"}
    assert session.history[-1] == committed
    assert session.last_assistant_full_message == {"role": "assistant", "content": "hello world"}


def test_history_commit_with_ack_only_policy_defers_unacknowledged_text():
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    response_id = session.begin_response()
    session.append_assistant_text("never played")
    session.mark_audio_sent(duration_ms=1_000, text_chars=12)

    committed = session.end_response(commit_text=True)

    assert committed is None
    assert session.history == ()
    assert f"item_{response_id}" in session.pending_history_item_ids


def test_truncate_history_item_uses_response_alignment_marks():
    session = _session()
    response_id = session.begin_response()
    session.append_assistant_text("hello ")
    session.mark_audio_sent(duration_ms=1_000, text_chars=6)
    session.append_assistant_text("world")
    session.mark_audio_sent(duration_ms=2_000, text_chars=11)
    session.acknowledge_playback(played_ms=2_000)
    committed = session.end_response(commit_text=True)
    assert committed == {"role": "assistant", "content": "hello world"}
    item_id = f"item_{response_id}"
    session.register_history_item(item_id, committed)

    assert session.truncate_history_item(item_id, audio_end_ms=1_000) is True

    assert session.history[-1] == {"role": "assistant", "content": "hello"}
    assert session.history_item_ids[item_id] == {"role": "assistant", "content": "hello"}
    assert session.delete_history_item(item_id) is True
    assert session.history == ()


# ---- turn signals / model turns ----


@pytest.mark.parametrize(
    ("event_type", "turn_state"),
    [
        (DuplexTurnEventType.USER_STARTED, DuplexTurnState.USER_SPEAKING),
        (DuplexTurnEventType.USER_COMMITTED, DuplexTurnState.USER_COMMITTED),
        (DuplexTurnEventType.ASSISTANT_STARTED, DuplexTurnState.ASSISTANT_GENERATING),
        (DuplexTurnEventType.ASSISTANT_DONE, DuplexTurnState.IDLE),
        (DuplexTurnEventType.BARGE_IN, DuplexTurnState.BARGE_IN),
    ],
)
def test_signal_turn_transitions_and_returns_typed_turn_event(event_type, turn_state):
    session = _session()

    event = session.signal_turn(event_type.value)

    assert isinstance(event, TurnEvent)
    assert session.turn_state == turn_state
    wire = event.to_realtime()
    assert wire["type"] == "turn.event"
    assert wire["event"] == event_type.value
    assert wire["turn_state"] == turn_state.value


def test_signal_turn_playback_ack_updates_cursor_and_close_marks_session_closing():
    session = _session()
    session.mark_audio_sent(duration_ms=5_000)

    ack = session.signal_turn("playback_ack", {"played_ms": 1_500, "committed_ms": 1_000})
    assert ack.event == "playback_ack"
    assert session.playback.played_ms == 1_500
    assert session.playback.committed_ms == 1_000

    close = session.signal_turn(DuplexTurnEventType.CLOSE.value)
    assert close.event == "close"
    assert session.state.value == "closing"


def test_complete_model_turn_advances_turn_id_and_fence():
    session = _session()
    assert session.fence == DuplexFence(session.session_id, epoch=0, turn_id=0)

    session.complete_model_turn(0)

    assert session.turn_id == 1
    assert session.fence == DuplexFence(session.session_id, epoch=0, turn_id=1)
    assert session.accepted_fence == session.fence

    # A stale terminal for an already-completed turn does not move identity backwards.
    session.complete_model_turn(0)
    assert session.turn_id == 1
    session.complete_model_turn(3)
    assert session.turn_id == 4
    assert session.accepted_fence.turn_id == 4


# ---- fence validation ----


def test_accept_fence_rejects_stale_or_foreign_fences():
    session = _session("sid-fence")
    session.complete_model_turn(0)

    with pytest.raises(DuplexFenceMismatchError):
        session.accept_fence(DuplexFence("sid-fence", epoch=0, turn_id=0))
    with pytest.raises(DuplexFenceMismatchError):
        session.accept_fence(DuplexFence("sid-other", epoch=0, turn_id=1))

    session.accept_fence(DuplexFence("sid-fence", epoch=1, turn_id=0))
    assert session.accepted_fence == DuplexFence("sid-fence", epoch=1, turn_id=0)


def test_prepare_and_commit_append_sequence_chunks_per_turn_and_epoch():
    session = _session("sid-append")
    fence = session.fence

    first = session.commit_append(session.prepare_append(fence))
    second = session.commit_append(session.prepare_append(fence))
    assert (first.seq, first.turn_seq, first.turn_id) == (1, 1, 0)
    assert (second.seq, second.turn_seq, second.turn_id) == (2, 2, 0)

    session.complete_model_turn(0)
    third = session.commit_append(session.prepare_append(session.fence))
    assert (third.seq, third.turn_seq, third.turn_id) == (3, 1, 1)

    # Barge-in advances the epoch (turn_id is kept) and restarts the append sequence.
    session.barge_in()
    assert session.fence == DuplexFence("sid-append", epoch=1, turn_id=1)
    fourth = session.commit_append(session.prepare_append(session.fence))
    assert (fourth.seq, fourth.turn_seq, fourth.turn_id) == (1, 1, 1)
    assert session.input_seq == 1


def test_commit_append_rejects_stale_reservation():
    session = _session("sid-append-stale")
    fence = session.fence
    reservation = session.prepare_append(fence)
    session.commit_append(session.prepare_append(fence))

    with pytest.raises(RuntimeError, match="stale"):
        session.commit_append(reservation)

    with pytest.raises(DuplexFenceMismatchError):
        session.prepare_append(DuplexFence("sid-other", epoch=0, turn_id=0))


def test_cancel_fence_releases_stage_requests_and_advances_identity():
    session = _session("sid-cancel")
    cancelled = session.fence
    session.reserve_stage_request(0, "req-a", fence=cancelled)
    session.bind_stage_request(1, "req-b", fence=cancelled)
    assert session.stage_request_submitted(1, "req-b") is True
    assert session.stage_request_submitted(0, "req-a") is False
    next_fence = DuplexFence("sid-cancel", epoch=1, turn_id=0)

    stale = session.cancel_fence(cancelled, next_fence)

    assert stale == ["req-a", "req-b"]
    assert session.resource_request_ids() == []
    assert session.accepted_fence == next_fence

    with pytest.raises(DuplexFenceMismatchError):
        session.cancel_fence(next_fence, DuplexFence("sid-cancel", epoch=1, turn_id=1))
    with pytest.raises(DuplexFenceMismatchError):
        session.cancel_fence(DuplexFence("sid-other", epoch=1, turn_id=0), DuplexFence("sid-cancel", epoch=2))


# ---- public view ----


def test_as_public_dict_exposes_identity_capabilities_and_playback():
    session = _session("duplex-public", config=DuplexSessionConfig(model="m", voice="alloy"))
    session.mark_audio_sent(duration_ms=100)

    payload = session.as_public_dict()

    assert payload["id"] == "duplex-public"
    assert payload["model"] == "m"
    assert payload["voice"] == "alloy"
    assert payload["state"] == "open"
    assert payload["turn_state"] == "assistant_playing"
    assert payload["epoch"] == 0
    assert payload["turn_id"] == 0
    assert payload["capabilities"] == DuplexCapabilities().as_dict()
    assert payload["playback"] == {"generated_ms": 100, "sent_ms": 100, "played_ms": 0, "committed_ms": 0}


# ---- DuplexSessionConfig.from_realtime ----


def test_from_realtime_rejects_unsupported_audio_formats():
    with pytest.raises(DuplexConfigError) as excinfo:
        DuplexSessionConfig.from_realtime({"input_audio_format": "mp3"})
    assert excinfo.value.code == "unsupported_audio_format"

    with pytest.raises(DuplexConfigError) as excinfo:
        DuplexSessionConfig.from_realtime({"audio": {"output": {"format": {"type": "audio/opus"}}}})
    assert excinfo.value.code == "unsupported_audio_format"


def test_from_realtime_validates_turn_detection():
    with pytest.raises(DuplexConfigError) as excinfo:
        DuplexSessionConfig.from_realtime({"turn_detection": {"type": "semantic_vad"}})
    assert excinfo.value.code == "unsupported_turn_detection"
    assert excinfo.value.param == "turn_detection"

    with pytest.raises(DuplexConfigError) as excinfo:
        DuplexSessionConfig.from_realtime({"overlap_policy": "barge_in_on_speech"})
    assert excinfo.value.code == "unsupported_turn_detection"

    server_vad = DuplexSessionConfig.from_realtime({"turn_detection": {"type": "server_vad"}})
    assert server_vad.overlap_policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value
    assert server_vad.extra_body["realtime_turn_detection"]["threshold"] == 0.5

    model_owned = DuplexSessionConfig.from_realtime({"turn_detection": None})
    assert model_owned.overlap_policy == DuplexOverlapPolicy.LISTEN_ONLY.value
    assert model_owned.extra_body["realtime_turn_detection"] is None


def test_from_realtime_maps_wire_fields_and_stores_realtime_keys_in_extra_body():
    config = DuplexSessionConfig.from_realtime(
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "instructions": "be brief",
            "output_audio_format": "pcm16",
            "max_response_output_tokens": "inf",
            "audio": {"output": {"voice": "alloy", "speed": 1.25}},
            "tools": [{"type": "function", "name": "lookup"}],
            "tool_choice": "auto",
            "metadata": {"tenant": "t1"},
            "include": ["item.input_audio_transcription.logprobs"],
            "input_audio_transcription": {"model": "whisper-1"},
            "extra_body": {"custom": True},
        }
    )

    assert config.model == "openbmb/MiniCPM-o-4_5"
    assert config.instructions == "be brief"
    assert config.voice == "alloy"
    assert config.speed == 1.25
    assert config.max_tokens is None
    assert config.response_format == "pcm"
    assert config.modalities == ["text", "audio"]
    assert config.extra_body["custom"] is True
    assert config.extra_body["realtime_tools"] == [{"type": "function", "name": "lookup"}]
    assert config.extra_body["realtime_tool_choice"] == "auto"
    assert config.extra_body["realtime_metadata"] == {"tenant": "t1"}
    assert config.extra_body["realtime_include"] == ["item.input_audio_transcription.logprobs"]
    assert config.extra_body["realtime_input_audio_transcription"] == {"model": "whisper-1"}
    assert config.extra_body["realtime_audio"] == {"output": {"voice": "alloy", "speed": 1.25}}
    assert config.extra_body["realtime_output_audio_format"] == "pcm16"
    assert "extra_body" not in config.extra_body["realtime_session_payload"]
    assert config.extra_body["realtime_session_payload"]["model"] == "openbmb/MiniCPM-o-4_5"


def test_from_realtime_uses_served_model_when_payload_has_none():
    config = DuplexSessionConfig.from_realtime({}, model="served-model")

    assert config.model == "served-model"
    assert config.response_format == "pcm"
    assert config.idle_timeout_s == 300.0


def test_from_realtime_ignores_client_chosen_session_ids():
    config = DuplexSessionConfig.from_realtime({"id": "client-chosen", "session_id": "also-client-chosen"})

    assert not hasattr(config, "session_id")
    assert "id" not in config.as_dict()
    session = _session("duplex-engine-allocated", config=config)
    assert session.as_public_dict()["id"] == "duplex-engine-allocated"


# ---- DuplexSessionConfig.apply_realtime_update ----


def test_apply_realtime_update_rejects_model_change():
    config = DuplexSessionConfig(model="served-model")

    with pytest.raises(DuplexConfigError) as excinfo:
        config.apply_realtime_update({"model": "other-model"}, session_id="sid")

    assert excinfo.value.code == "model_update_unsupported"
    assert config.model == "served-model"


def test_apply_realtime_update_rejects_voice_change_after_audio_started():
    config = DuplexSessionConfig(model="m", voice="alloy")

    config.apply_realtime_update({"voice": "verse"}, audio_started=False)
    assert config.voice == "verse"
    with pytest.raises(DuplexConfigError) as excinfo:
        config.apply_realtime_update({"audio": {"output": {"voice": "alloy"}}}, audio_started=True)
    assert excinfo.value.code == "voice_update_after_audio_unsupported"
    assert config.voice == "verse"


def test_apply_realtime_update_rejects_ref_audio_change():
    config = DuplexSessionConfig(model="m")

    with pytest.raises(DuplexConfigError) as excinfo:
        config.apply_realtime_update({"ref_audio": "/tmp/voice.wav"})

    assert excinfo.value.code == "ref_audio_update_unsupported"
    assert config.ref_audio is None


def test_apply_realtime_update_patches_fields_and_realtime_keys():
    config = DuplexSessionConfig(model="m", instructions="old", extra_body={"realtime_tools": [{"name": "x"}]})

    config.apply_realtime_update(
        {
            "instructions": "new",
            "output_audio_format": "g711_ulaw",
            "max_output_tokens": 32,
            "tools": None,
            "metadata": {"k": "v"},
            "overlap_policy": "not-a-policy",
        }
    )

    assert config.instructions == "new"
    assert config.response_format == "pcm"
    assert config.max_tokens == 32
    assert "realtime_tools" not in config.extra_body
    assert config.extra_body["realtime_metadata"] == {"k": "v"}
    assert config.overlap_policy == DuplexOverlapPolicy.LISTEN_ONLY.value
    assert config.extra_body["realtime_session_payload"]["instructions"] == "new"


# ---- overlap policy / capabilities ----


def test_overlap_policy_defaults_and_invalid_values_to_listen_only():
    assert DuplexSessionConfig().overlap_policy == DuplexOverlapPolicy.LISTEN_ONLY.value
    assert DuplexSessionConfig._normalize_overlap_policy("auto") == DuplexOverlapPolicy.LISTEN_ONLY.value
    assert DuplexSessionConfig._normalize_overlap_policy("not-a-policy") == DuplexOverlapPolicy.LISTEN_ONLY.value
    assert (
        DuplexSessionConfig._normalize_overlap_policy(" Barge_In_On_Speech ")
        == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value
    )
    assert DuplexSessionConfig.from_event({"session": {"overlap_policy": "bogus"}}).overlap_policy == "listen_only"


def test_capabilities_as_dict_reports_model_native_duplex_constants():
    caps = DuplexCapabilities().as_dict()

    assert caps["implementation_level"] == "model_native_duplex"
    assert caps["input_modes"] == ["append_audio_chunk"]
    assert caps["supports_kv_lease"] is False
    assert caps["supports_core_kv_lease"] is False
    assert "client_event" in caps["signal_sources"]
    assert set(caps) == set(minicpmo45_native_capabilities().as_dict())


def test_minicpmo_native_capabilities_separate_model_state_from_core_kv_lease():
    caps = minicpmo45_native_capabilities(max_sessions=2).as_dict()

    assert caps["implementation_level"] == "model_native_duplex"
    assert caps["supports_input_append"] is True
    assert caps["input_modes"] == ["append_audio_chunk"]
    assert caps["adapter_patterns"] == ["scheduler_data_plane"]
    assert caps["supports_model_internal_state"] is True
    assert caps["requires_model_runner_kv"] is True
    assert caps["requires_native_stage_role"] is True
    assert caps["supports_kv_lease"] is False
    assert caps["supports_core_kv_lease"] is False
    assert caps["supports_stage_resumption"] is True
    assert caps["supports_scheduler_native_append"] is False
    assert caps["supports_core_resumable_request"] is True
    assert caps["supports_stage_connector_handoff"] is True
    assert caps["supports_audio_truncate"] is True
    assert caps["supports_barge_in"] is True
    assert caps["target_barge_in_latency_ms"] is None
    assert caps["supports_multi_session"] is True
    assert caps["supports_multi_session_same_replica"] is True
    assert caps["supports_session_lease"] is True
    assert caps["supports_session_resume"] is True
    assert caps["session_admission_mode"] == "engine_managed"
    assert caps["stage_handoff_transport"] == "scheduler_data_plane"


def test_minicpmo_native_capabilities_do_not_overclaim_single_session_deployment():
    caps = minicpmo45_native_capabilities(max_sessions=1).as_dict()

    assert caps["supports_multi_session"] is False
    assert caps["supports_multi_session_same_replica"] is False
