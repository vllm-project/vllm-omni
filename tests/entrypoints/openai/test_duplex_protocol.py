import asyncio
import json

import pytest

from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexOverlapPolicy,
    DuplexSession,
    DuplexSessionConfig,
    DuplexSessionRegistry,
    DuplexTurnController,
    DuplexTurnEventType,
    DuplexTurnState,
    ResponseCreateOptions,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import (
    NativeRealtimeSessionProtocol,
)
from vllm_omni.experimental.fullduplex.openai.realtime_state import RealtimeStateOwner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_state_owner_uses_explicit_bindings_not_dynamic_attribute_proxy():
    assert "__getattr__" not in RealtimeStateOwner.__dict__
    assert "__setattr__" not in RealtimeStateOwner.__dict__


class _ProtocolWebSocket:
    def __init__(self, *events: dict[str, object]) -> None:
        self._events = asyncio.Queue()
        for event in events:
            self._events.put_nowait(json.dumps(event))

    async def receive_text(self) -> str:
        return await self._events.get()


def test_duplex_session_commits_text_and_audio_as_one_turn():
    registry = DuplexSessionRegistry()
    session = registry.create(DuplexSessionConfig(model="test-model"))

    session.append_text("hello")
    session.append_audio("YWJj", fmt="wav", sample_rate_hz=16000)
    committed = session.commit_user_input()

    assert committed is not None
    assert committed.turn_id == 0
    assert committed.input_commit_seq == 1
    assert committed.epoch == 0
    assert len(session.history) == 1
    content = session.history[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1]["type"] == "audio_url"
    assert content[1]["audio_url"]["url"] == "data:audio/wav;base64,YWJj"
    assert session.turn_state == DuplexTurnState.USER_COMMITTED


def test_response_options_apply_to_one_response_without_mutating_session_defaults():
    session = DuplexSession(
        session_id="sid-response-options",
        config=DuplexSessionConfig(instructions="base", voice="base-voice", max_tokens=64),
    )
    session.reserve_response_options(
        ResponseCreateOptions(
            instructions="one response",
            voice="override-voice",
            max_tokens=8,
        )
    )

    assert session.config.instructions == "base"
    assert session.config.voice == "base-voice"
    session.begin_response()
    assert session.config.instructions == "base"
    assert session.config.voice == "base-voice"
    assert session.response_config.instructions == "one response"
    assert session.response_config.voice == "override-voice"
    assert session.response_config.max_tokens == 8

    session.end_response()
    assert session.config.instructions == "base"
    assert session.config.voice == "base-voice"
    assert session.config.max_tokens == 64


def test_response_options_cannot_overwrite_an_unconsumed_reservation():
    session = DuplexSession(
        session_id="sid-response-options-pending",
        config=DuplexSessionConfig(instructions="base"),
    )
    session.reserve_response_options(ResponseCreateOptions(instructions="first"))

    with pytest.raises(RuntimeError, match="already reserved"):
        session.reserve_response_options(ResponseCreateOptions(instructions="second"))

    session.begin_response()
    assert session.response_config.instructions == "first"


def test_response_options_cannot_be_reserved_while_response_is_active():
    session = DuplexSession(
        session_id="sid-response-options-active",
        config=DuplexSessionConfig(instructions="base"),
    )
    session.begin_response()

    with pytest.raises(RuntimeError, match="active"):
        session.reserve_response_options(ResponseCreateOptions(instructions="too late"))


def test_realtime_model_name_does_not_implicitly_enable_native_duplex():
    protocol = NativeRealtimeSessionProtocol({})

    event = protocol._session_create_from_realtime({"model": "openbmb/MiniCPM-o-4_5"})

    assert "minicpmo45_native_duplex" not in event["session"]["extra_body"]


def test_realtime_explicit_native_duplex_flag_is_preserved():
    protocol = NativeRealtimeSessionProtocol({})

    event = protocol._session_create_from_realtime(
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "extra_body": {"minicpmo45_native_duplex": True},
        }
    )

    assert event["session"]["extra_body"]["minicpmo45_native_duplex"] is True


def test_realtime_explicit_query_native_duplex_flag_is_available_before_autostart():
    protocol = NativeRealtimeSessionProtocol(
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "minicpmo45_native_duplex": "1",
        }
    )

    event = json.loads(asyncio.run(protocol.receive_internal_event_text(None)))

    assert event["type"] == "session.create"
    assert event["session"]["extra_body"]["minicpmo45_native_duplex"] is True


@pytest.mark.asyncio
async def test_realtime_resume_heartbeat_and_event_ack_translate_without_session_autostart():
    protocol = NativeRealtimeSessionProtocol(
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "minicpmo45_native_duplex": "1",
        }
    )

    resume = await protocol._to_duplex_event(
        {
            "type": "session.resume",
            "session_id": "sid-resume",
            "incarnation": 3,
            "resume_token": "opaque-token",
            "last_received_server_event_seq": 17,
        }
    )
    heartbeat = await protocol._to_duplex_event(
        {
            "type": "session.heartbeat",
            "event_id": "event-heartbeat",
        }
    )
    event_ack = await protocol._to_duplex_event(
        {
            "type": "session.event_ack",
            "server_event_seq": 19,
        }
    )

    assert resume == {
        "type": "session.resume",
        "session_id": "sid-resume",
        "incarnation": 3,
        "resume_token": "opaque-token",
        "last_received_server_event_seq": 17,
    }
    assert heartbeat == {
        "type": "session.heartbeat",
        "event_id": "event-heartbeat",
    }
    assert event_ack == {
        "type": "session.event_ack",
        "server_event_seq": 19,
    }
    assert protocol._opened is False

    wire_protocol = NativeRealtimeSessionProtocol({})
    first_wire_event = json.loads(
        await wire_protocol.receive_internal_event_text(
            _ProtocolWebSocket(
                {
                    "type": "session.resume",
                    "session_id": "sid-resume",
                    "incarnation": 3,
                    "resume_token": "opaque-token",
                    "last_received_server_event_seq": 17,
                }
            )
        )
    )

    assert first_wire_event["type"] == "session.resume"


@pytest.mark.asyncio
async def test_realtime_resume_query_suppresses_model_autostart_on_same_url():
    protocol = NativeRealtimeSessionProtocol(
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "minicpmo45_native_duplex": "1",
            "resume": "1",
        }
    )
    resume = {
        "type": "session.resume",
        "session_id": "sid-resume",
        "incarnation": 3,
        "resume_token": "opaque-token",
        "last_received_server_event_seq": 17,
    }

    first_wire_event = json.loads(await protocol.receive_internal_event_text(_ProtocolWebSocket(resume)))

    assert first_wire_event == resume
    assert protocol._autostarted_default_session is False


def test_realtime_projects_resume_lifecycle_events_without_duplex_prefix():
    protocol = NativeRealtimeSessionProtocol({})

    created = protocol.encode_outbound_event(
        {
            "type": "session.created",
            "session": {"id": "sid-resume"},
            "incarnation": 3,
            "attachment_generation": 1,
            "resume_token": "first-token",
        }
    )[0]
    resumed = protocol.encode_outbound_event(
        {
            "type": "session.resumed",
            "session_id": "sid-resume",
            "incarnation": 3,
            "attachment_generation": 2,
            "resume_token": "second-token",
        }
    )[0]
    heartbeat = protocol.encode_outbound_event(
        {
            "type": "session.heartbeat_ack",
            "session_id": "sid-resume",
        }
    )[0]
    replaced = protocol.encode_outbound_event(
        {
            "type": "session.replaced",
            "session_id": "sid-resume",
            "attachment_generation": 1,
        }
    )[0]
    resync = protocol.encode_outbound_event(
        {
            "type": "session.resync_required",
            "session_id": "sid-resume",
            "reason": "journal_gap",
        }
    )[0]

    assert created["type"] == "session.created"
    assert created["incarnation"] == 3
    assert created["attachment_generation"] == 1
    assert created["resume_token"] == "first-token"
    assert resumed["type"] == "session.resumed"
    assert resumed["resume_token"] == "second-token"
    assert heartbeat["type"] == "session.heartbeat_ack"
    assert replaced["type"] == "session.replaced"
    assert resync == {
        "type": "session.resync_required",
        "session_id": "sid-resume",
        "reason": "journal_gap",
        "event_id": resync["event_id"],
    }


def test_duplex_session_registry_advances_incarnation_when_id_is_reused():
    registry = DuplexSessionRegistry()

    first = registry.create(session_id="reused-session")
    registry.close(first.session_id)
    second = registry.create(session_id="reused-session")

    assert first.incarnation == 0
    assert second.incarnation == 1


def test_native_input_commit_does_not_advance_model_turn_identity():
    session = DuplexSession(
        session_id="sid-model-owned-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )

    first = session.commit_native_audio_input(transcript="first chunk")
    second = session.commit_native_audio_input(transcript="second chunk")

    assert session.input_commit_seq == 2
    assert first.input_commit_seq == 1
    assert second.input_commit_seq == 2
    assert first.turn_id == second.turn_id == session.turn_id == 0


def test_duplex_session_owns_response_and_overlap_identity():
    session = DuplexSession(session_id="sid-state-owner", config=DuplexSessionConfig())

    response_id = session.begin_response()
    session.accumulate_overlap_speech(320)
    session.accumulate_overlap_speech(180)
    session.end_response()

    assert session.active_response_id is None
    assert session.last_response_id == response_id
    assert session.overlap_speech_ms == 500
    assert session.reset_overlap_speech() == 500
    assert session.overlap_speech_ms == 0


def test_duplex_session_composes_single_owner_ledgers():
    session = DuplexSession(session_id="sid-ledgers", config=DuplexSessionConfig())

    session.append_text("hello")
    session.bind_request("req-1")
    response_id = session.begin_response(turn_id=3)
    session.mark_audio_sent(duration_ms=240)

    assert not hasattr(session, "input")
    assert not hasattr(session, "response")
    assert not hasattr(session, "playback_ledger")
    assert not hasattr(session, "conversation")
    assert session.pending_text == ("hello",)
    assert session.active_request_id == "req-1"
    assert session.active_response_id == response_id
    assert session.active_response_turn_id == 3
    assert session.playback.sent_ms == 240

    try:
        session.playback.sent_ms = 480
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("playback view must be immutable")


def test_duplex_barge_in_advances_epoch_and_drops_uncommitted_assistant_text():
    registry = DuplexSessionRegistry()
    session = registry.create()
    response_id = session.begin_response()
    session.bind_request("chatcmpl-duplex-test")
    session.append_assistant_text("unplayed answer")

    new_epoch = session.barge_in()

    assert response_id is not None
    assert new_epoch == 1
    assert session.epoch == 1
    assert session.active_request_id is None
    assert session.active_response_id is None
    assert session.assistant_text_buffer == ()
    assert session.history == ()
    assert session.turn_state == DuplexTurnState.BARGE_IN


def test_duplex_playback_ack_tracks_committed_cursor_separately():
    registry = DuplexSessionRegistry()
    session = registry.create()
    session.mark_audio_sent(duration_ms=10_000)

    session.acknowledge_playback(played_ms=2_000)

    assert session.playback.generated_ms == 10_000
    assert session.playback.sent_ms == 10_000
    assert session.playback.played_ms == 2_000
    assert session.playback.committed_ms == 2_000


def test_duplex_history_commit_uses_audio_text_alignment_marks():
    registry = DuplexSessionRegistry()
    session = registry.create()
    session.begin_response()

    session.append_assistant_text("hello ")
    session.mark_audio_sent(duration_ms=1_000, text_chars=6)
    session.append_assistant_text("world")
    session.mark_audio_sent(duration_ms=2_000, text_chars=11)
    session.acknowledge_playback(played_ms=1_200, committed_ms=1_200)

    committed = session.end_response(commit_text=True)

    assert committed == {"role": "assistant", "content": "hello w"}


def test_turn_controller_accepts_external_signal_source():
    registry = DuplexSessionRegistry()
    session = registry.create()
    controller = DuplexTurnController()

    event = controller.signal(session, DuplexTurnEventType.USER_STARTED.value)

    assert event["type"] == "turn.event"
    assert event["event"] == "user_started"
    assert event["turn_state"] == "user_speaking"
    assert session.turn_state == DuplexTurnState.USER_SPEAKING


def test_duplex_capabilities_do_not_claim_core_kv_or_input_append():
    registry = DuplexSessionRegistry()
    session = registry.create()
    caps = session.capabilities.as_dict()

    assert caps["implementation_level"] == "serving_session_adapter"
    assert caps["supports_kv_lease"] is False
    assert caps["supports_input_append"] is False
    assert caps["supports_reencode_context"] is True
    assert caps["adapter_patterns"] == ["chunk_group_append"]
    assert "turn_commit_only" in caps["input_modes"]
    assert "client_event" in caps["signal_sources"]


def test_minicpmo_native_capabilities_separate_model_state_from_core_kv_lease():
    caps = DuplexCapabilities.minicpmo45_native(max_sessions=2).as_dict()

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
    assert caps["supports_barge_in"] is False
    assert caps["target_barge_in_latency_ms"] is None
    assert caps["supports_multi_session"] is True
    assert caps["supports_multi_session_same_replica"] is True
    assert caps["supports_session_lease"] is True
    assert caps["supports_session_resume"] is True
    assert caps["session_admission_mode"] == "engine_managed"
    assert caps["stage_handoff_transport"] == "scheduler_data_plane"


def test_minicpmo_native_capabilities_do_not_overclaim_single_session_deployment():
    caps = DuplexCapabilities.minicpmo45_native(max_sessions=1).as_dict()

    assert caps["supports_multi_session"] is False
    assert caps["supports_multi_session_same_replica"] is False


def test_duplex_overlap_policy_defaults_and_invalid_values_to_listen_only():
    assert DuplexSessionConfig().overlap_policy == DuplexOverlapPolicy.LISTEN_ONLY.value
    assert DuplexSessionConfig._normalize_overlap_policy("not-a-policy") == DuplexOverlapPolicy.LISTEN_ONLY.value
