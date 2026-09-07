# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the public duplex client (fake transport, no server)."""

from __future__ import annotations

import asyncio
import base64
import json
import wave
from urllib.parse import parse_qs, urlsplit

import pytest

from vllm_omni.clients.duplex import (
    AudioFormat,
    ConnectionResumed,
    DuplexClient,
    DuplexProtocolError,
    DuplexSessionClosedError,
    EventCollector,
    ReconnectPolicy,
    SessionConfig,
    SessionResumed,
    build_realtime_url,
    chunk_period_ms,
    duplex_unit_boundary_ms,
    has_residual_model_unit,
    read_pcm16_wav,
    reference_audio_data_url,
    summarize_session_request_metrics,
    write_pcm16_wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_CREATED = {
    "type": "session.created",
    "session": {"session_id": "sess-1"},
    "incarnation": 0,
    "resume_token": "tok-1",
    "server_event_seq": 1,
}
SESSION_CLOSED = {"type": "session.closed", "session_id": "sess-1", "server_event_seq": 99}


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, raw: str) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)

    async def close(self) -> None:
        self.closed = True

    def feed(self, event: dict[str, object] | BaseException) -> None:
        self.incoming.put_nowait(event)

    def sent_types(self) -> list[object]:
        return [event.get("type") for event in self.sent]


def make_client(*sockets: FakeSocket, **kwargs) -> tuple[DuplexClient, list[str]]:
    remaining = list(sockets)
    calls: list[str] = []

    async def connect(url: str):
        calls.append(url)
        if not remaining:
            raise ConnectionError("no more sockets")
        return remaining.pop(0)

    kwargs.setdefault("heartbeat_interval_s", None)
    kwargs.setdefault("reconnect", None)
    kwargs.setdefault("handshake_timeout_s", 5.0)
    client = DuplexClient("ws://test-host:8099", model="test-model", connect=connect, **kwargs)
    return client, calls


# ---------------------------------------------------------------------------
# SessionConfig


def test_session_config_payload_defaults():
    payload = SessionConfig().to_session_payload(model="m", session_id="s")
    assert payload["model"] == "m"
    assert payload["session_id"] == "s"
    assert payload["modalities"] == ["audio", "text"]
    assert payload["input_audio_format"] == "pcm16"
    assert payload["output_audio_format"] == "pcm16"
    # The encodings do not pin the rates; the payload must carry both (the
    # server reads the input rate top-level and the output rate from audio).
    assert payload["sample_rate_hz"] == 16_000
    assert payload["audio"] == {
        "input": {"sample_rate_hz": 16_000},
        "output": {"sample_rate_hz": 24_000},
    }
    assert payload["turn_detection"] is None
    assert payload["extra_body"] == {"auto_response": True}
    assert "voice" not in payload
    assert "ref_audio" not in payload


def test_session_config_payload_carries_preset_rates():
    config = SessionConfig(
        input_audio=AudioFormat("pcm_f32le", 24_000),
        output_audio=AudioFormat("pcm16", 24_000),
    )
    payload = config.to_session_payload(model="m")
    assert payload["input_audio_format"] == "pcm_f32le"
    assert payload["sample_rate_hz"] == 24_000
    assert payload["audio"] == {
        "input": {"sample_rate_hz": 24_000},
        "output": {"sample_rate_hz": 24_000},
    }


def test_audio_format_math():
    fmt = AudioFormat("pcm16", 16_000)
    assert fmt.byte_count(100) == 3200
    assert fmt.duration_ms(3200) == 100.0
    f32 = AudioFormat("pcm_f32le", 24_000)
    assert f32.byte_count(80) == 1920 * 4
    with pytest.raises(ValueError):
        _ = AudioFormat("mp3", 16_000).bytes_per_sample


# ---------------------------------------------------------------------------
# Handshake and lifecycle


async def test_handshake_adopts_session_and_acks():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, calls = make_client(sock)
    async with client:
        assert client.session_id == "sess-1"
        assert client.resume_token == "tok-1"
        assert calls == ["ws://test-host:8099/v1/realtime?duplex=1&model=test-model&autostart=0"]
        assert sock.sent[0]["type"] == "session.update"
        assert sock.sent[0]["session"]["model"] == "test-model"
        await _drain(lambda: {"type": "session.event_ack", "server_event_seq": 1} in _acks(sock))
        close_task = asyncio.create_task(client.close())
        await _drain(lambda: "session.close" in sock.sent_types())
        sock.feed(SESSION_CLOSED)
        await close_task
    assert "session.close" in sock.sent_types()


async def test_handshake_error_raises_protocol_error():
    sock = FakeSocket()
    sock.feed({"type": "error", "error": {"code": "unsupported_audio_format", "message": "bad"}})
    client, _ = make_client(sock)
    with pytest.raises(DuplexProtocolError) as excinfo:
        async with client:
            pass
    assert excinfo.value.code == "unsupported_audio_format"


async def test_session_expired_raises_from_event_stream():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "session.expired", "reason": "lease_expired", "server_event_seq": 2})
        with pytest.raises(DuplexSessionClosedError) as excinfo:
            async for _ in client.events():
                pass
        assert "lease_expired" in excinfo.value.reason


async def test_resync_required_drops_resume_credential():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, calls = make_client(sock, reconnect=ReconnectPolicy(max_attempts=2, backoff_s=(0.0, 0.0)))
    async with client:
        assert client.resume_token == "tok-1"
        stream = client.events()
        sock.feed({"type": "session.resync_required", "session_id": "sess-1", "server_event_seq": 2})
        async for event in stream:
            if event.type == "session.resync_required":
                break
        assert client.resume_token is None
        # With the credential gone, a transport drop must finalize instead of
        # attempting session.resume against a server that stopped journaling.
        sock.feed(RuntimeError("transport dropped"))
        with pytest.raises(DuplexSessionClosedError):
            async for _ in stream:
                pass
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Input events


async def test_append_audio_tracks_cumulative_end_ms():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        pcm = b"\x01\x02" * 1600  # 100 ms of pcm16 @ 16 kHz
        await client.append_audio(pcm)
        await client.append_audio(pcm, is_speech=True)
        appends = [event for event in sock.sent if event.get("type") == "input_audio_buffer.append"]
        assert [a["audio_end_ms"] for a in appends] == [100, 200]
        assert appends[0]["format"] == "pcm16"
        assert appends[0]["sample_rate_hz"] == 16_000
        assert appends[0]["duration_ms"] == 100
        assert "is_speech" not in appends[0]
        assert appends[1]["is_speech"] is True
        assert base64.b64decode(appends[0]["audio"]) == pcm
        sock.feed(SESSION_CLOSED)


async def test_append_audio_strips_video_frame_data_url_prefix():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        jpeg_b64 = base64.b64encode(b"\xff\xd8fake").decode("ascii")
        # The wire contract carries bare base64; image_data_url output must
        # still be accepted (the server validator rejects data-URL prefixes).
        await client.append_audio(b"\x00\x00", video_frames=[f"data:image/jpeg;base64,{jpeg_b64}", jpeg_b64])
        append = next(event for event in sock.sent if event.get("type") == "input_audio_buffer.append")
        assert append["video_frames"] == [jpeg_b64, jpeg_b64]
        sock.feed(SESSION_CLOSED)


async def test_stream_pcm_chunking():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        await client.stream_pcm(b"\x00" * 8000, chunk_ms=100, realtime=False)
        appends = [event for event in sock.sent if event.get("type") == "input_audio_buffer.append"]
        assert [a["duration_ms"] for a in appends] == [100, 100, 50]
        sock.feed(SESSION_CLOSED)


async def test_interruption_primitives_send_documented_events():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        await client.cancel_response()
        await client.clear_input()
        types = sock.sent_types()
        assert types.index("response.cancel") < types.index("input_audio_buffer.clear")
        sock.feed(SESSION_CLOSED)


# ---------------------------------------------------------------------------
# Response demultiplexing


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


async def test_response_handle_flow():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    chunk = b"\x00\x01" * 2400  # 100 ms of pcm16 @ 24 kHz
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        sock.feed({"type": "response.speak", "response_id": "resp-1", "server_event_seq": 3})
        sock.feed(
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "delta": _b64(chunk),
                "sample_rate_hz": 24_000,
                "server_event_seq": 4,
            }
        )
        sock.feed(
            {
                "type": "response.audio_transcript.delta",
                "response_id": "resp-1",
                "delta": "hi there",
                "server_event_seq": 5,
            }
        )
        sock.feed({"type": "response.done", "response_id": "resp-1", "server_event_seq": 6})

        async for response in client.responses():
            chunks = [piece async for piece in response.audio()]
            assert chunks == [chunk]
            assert response.decision == "speak"
            assert response.transcript == "hi there"
            assert response.played_ms == pytest.approx(100.0)
            assert response.finished
            break
        sock.feed(SESSION_CLOSED)


async def test_listen_decision_yields_finished_silent_handle():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.listen", "server_event_seq": 2})
        async for response in client.responses():
            assert response.decision == "listen"
            assert response.finished
            assert [piece async for piece in response.audio()] == []
            break
        sock.feed(SESSION_CLOSED)


async def test_listen_terminates_active_response():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        sock.feed({"type": "response.listen", "response_id": "resp-1", "server_event_seq": 3})
        async for response in client.responses():
            await response.wait(timeout_s=5.0)
            assert response.decision == "listen"
            break
        sock.feed(SESSION_CLOSED)


async def test_id_less_listen_never_closes_an_in_flight_response():
    # An id-less listen is a standalone decision beat (e.g. a silence-skip
    # while a response streams); it must surface as its own finished handle
    # and leave the in-flight response to its real terminal.
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        sock.feed({"type": "response.speak", "response_id": "resp-1", "server_event_seq": 3})
        sock.feed({"type": "response.listen", "server_event_seq": 4})
        sock.feed({"type": "response.done", "response_id": "resp-1", "server_event_seq": 5})
        seen: list[tuple[str | None, str | None, bool]] = []
        async for response in client.responses():
            await response.wait(timeout_s=5.0)
            seen.append((response.response_id, response.decision, response.finished))
            if len(seen) == 2:
                break
        assert ("resp-1", "speak", True) in seen
        assert (None, "listen", True) in seen
        sock.feed(SESSION_CLOSED)


async def test_terminal_listen_keeps_spoken_decision():
    # A terminal listen after the response spoke is the model yielding the
    # turn; the handle must stay decision="speak" with its audio intact.
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    chunk = b"\x00\x01" * 2400
    client, _ = make_client(sock)
    async with client:
        sock.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        sock.feed({"type": "response.speak", "response_id": "resp-1", "server_event_seq": 3})
        sock.feed(
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "delta": _b64(chunk),
                "sample_rate_hz": 24_000,
                "server_event_seq": 4,
            }
        )
        sock.feed({"type": "response.listen", "response_id": "resp-1", "server_event_seq": 5})
        async for response in client.responses():
            chunks = [piece async for piece in response.audio()]
            assert chunks == [chunk]
            assert response.decision == "speak"
            assert response.finished
            break
        sock.feed(SESSION_CLOSED)


async def test_error_event_surfaces_on_response_path():
    # A rejected send produces no response; a consumer waiting in responses()
    # must see the rejection instead of waiting forever.
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        sock.feed(
            {
                "type": "error",
                "error": {"code": "input_audio_buffer_empty", "message": "empty commit"},
                "server_event_seq": 2,
            }
        )
        with pytest.raises(DuplexProtocolError) as excinfo:
            async for _ in client.responses():
                pass
        assert excinfo.value.code == "input_audio_buffer_empty"
        sock.feed(SESSION_CLOSED)


async def test_slow_audio_consumer_drops_oldest_instead_of_stalling():
    from vllm_omni.clients.duplex import AudioDelta, ResponseHandle

    handle = ResponseHandle(
        response_id="r1",
        output_format=AudioFormat("pcm16", 24_000),
        max_buffered_events=2,
    )
    for payload in (b"c1", b"c2", b"c3"):
        # _feed must never block the reader, even against a full queue.
        handle._feed(AudioDelta({"type": "response.audio.delta", "delta": _b64(payload)}))
    handle._finish(None)
    chunks = [chunk async for chunk in handle.audio()]
    assert chunks == [b"c3"]  # oldest chunks were dropped, the sentinel landed


# ---------------------------------------------------------------------------
# Resume


async def test_resume_after_transport_drop():
    first = FakeSocket()
    first.feed(SESSION_CREATED)
    second = FakeSocket()
    # Mirror the real resume activation payload
    # (entrypoints/duplex/serving.py, activation_payload_factory): it carries
    # no nested "session" object, only the identity fields.
    second.feed(
        {
            "type": "session.resumed",
            "session_id": "sess-1",
            "incarnation": 0,
            "attachment_generation": 1,
            "resume_token": "tok-2",
            "server_event_seq": 5,
        }
    )
    client, calls = make_client(
        first,
        second,
        reconnect=ReconnectPolicy(max_attempts=2, backoff_s=(0.0, 0.0)),
    )
    async with client:
        stream = client.events()

        async def take_two():
            seen = []
            async for event in stream:
                seen.append(event)
                if len(seen) == 2:
                    return seen

        consumer = asyncio.create_task(take_two())
        await asyncio.sleep(0)  # let the consumer subscribe before the drop
        await asyncio.sleep(0)
        first.feed(RuntimeError("transport dropped"))
        seen = await asyncio.wait_for(consumer, timeout=5.0)
        assert isinstance(seen[0], ConnectionResumed)
        assert isinstance(seen[1], SessionResumed)
        assert client.resume_token == "tok-2"
        # The activation payload has no session object; the info captured at
        # the original handshake must survive the resume.
        assert client.session_info == {"session_id": "sess-1"}
        assert len(calls) == 2
        resume = second.sent[0]
        assert resume["type"] == "session.resume"
        assert resume["session_id"] == "sess-1"
        assert resume["incarnation"] == 0
        assert resume["resume_token"] == "tok-1"
        assert resume["last_received_server_event_seq"] == 1
        second.feed(SESSION_CLOSED)


async def test_no_reconnect_policy_surfaces_closed():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)  # reconnect=None
    async with client:
        stream = client.events()
        sock.feed(RuntimeError("transport dropped"))
        with pytest.raises(DuplexSessionClosedError):
            async for _ in stream:
                pass


async def test_heartbeat_survives_transient_send_failure():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    armed = {"fail": True}
    original_send = sock.send

    async def flaky_send(raw: str) -> None:
        if json.loads(raw).get("type") == "session.heartbeat" and armed["fail"]:
            armed["fail"] = False
            raise RuntimeError("transport hiccup")
        await original_send(raw)

    sock.send = flaky_send  # type: ignore[method-assign]
    client, _ = make_client(sock, heartbeat_interval_s=0.01)
    async with client:
        # The first tick fails; the loop must keep ticking (a resumed but
        # quiet session relies on heartbeats to reset the server timeout).
        await _drain(lambda: "session.heartbeat" in sock.sent_types())
        sock.feed(SESSION_CLOSED)


async def test_aexit_sends_session_close_on_error_path():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    with pytest.raises(RuntimeError):
        async with client:
            raise RuntimeError("application failure")
    # The server must not be left holding the session for the disconnect
    # grace period; the error path still announces the close.
    assert "session.close" in sock.sent_types()


def test_target_url_forces_autostart_off():
    client = DuplexClient("ws://test-host:8099/v1/realtime?autostart=1", model="m")
    query = parse_qs(urlsplit(client._target_url()).query)
    # autostart would race the session.update handshake and silently drop
    # ref_audio/extra_body; the client overrides it even when the URL asks.
    assert query["autostart"] == ["0"]


async def test_fatal_resume_error_gives_up():
    first = FakeSocket()
    first.feed(SESSION_CREATED)
    second = FakeSocket()
    second.feed({"type": "error", "error": {"code": "invalid_resume_token", "message": "nope"}})
    client, calls = make_client(
        first,
        second,
        reconnect=ReconnectPolicy(max_attempts=3, backoff_s=(0.0, 0.0)),
    )
    async with client:
        stream = client.events()
        first.feed(RuntimeError("transport dropped"))
        with pytest.raises(DuplexSessionClosedError):
            async for _ in stream:
                pass
        assert len(calls) == 2  # no retry after a fatal resume error


# ---------------------------------------------------------------------------
# Collector


def test_event_collector_accumulates_audio():
    collector = EventCollector()
    collector.add({"type": "response.created", "response": {"id": "r1"}}, received_at_s=1.0)
    collector.add(
        {"type": "response.audio.delta", "response_id": "r1", "delta": _b64(b"ab"), "sample_rate_hz": 16_000},
        received_at_s=1.1,
    )
    collector.add(
        {"type": "response.output_audio.delta", "response_id": "r1", "audio": _b64(b"cd")},
        received_at_s=1.2,
    )
    assert collector.count("response.created") == 1
    assert collector.audio_bytes() == b"abcd"
    assert collector.output_sample_rate_hz == 16_000
    summary = collector.timing_summary(after_s=0.0)
    assert summary["audio_output"]["chunk_count"] == 2
    assert summary["audio_output"]["response_created_to_first_audio_ms"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# helpers


def _acks(sock: FakeSocket) -> list[dict[str, object]]:
    return [event for event in sock.sent if event.get("type") == "session.event_ack"]


async def _drain(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Probe/benchmark helpers (ported from the retired MiniCPM demo client tests)


def test_build_realtime_url_with_model_extra_query():
    url = build_realtime_url(
        "ws://localhost:8099/v1/realtime?custom=1",
        "openbmb/MiniCPM-o-4_5",
        session_id="session-a",
        extra_query={"native_duplex": "1"},
    )

    query = parse_qs(urlsplit(url).query)
    assert query == {
        "custom": ["1"],
        "duplex": ["1"],
        "model": ["openbmb/MiniCPM-o-4_5"],
        "native_duplex": ["1"],
        "session_id": ["session-a"],
    }


def test_build_realtime_url_resume_only_when_autostart_disabled():
    url = build_realtime_url(
        "ws://localhost:8099/v1/realtime?duplex=1",
        "openbmb/MiniCPM-o-4_5",
        autostart=False,
        extra_query={"native_duplex": "1"},
    )

    query = parse_qs(urlsplit(url).query)
    assert query["autostart"] == ["0"]
    assert query["native_duplex"] == ["1"]


def test_event_collector_partitions_audio_by_response():
    collector = EventCollector()
    collector.add({"type": "response.created", "response": {"id": "resp-a"}})
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(b"audio-a").decode("ascii"),
            "sample_rate_hz": 16_000,
        }
    )

    assert collector.response_ids == ["resp-a"]
    assert collector.audio_bytes("resp-a") == b"audio-a"
    assert collector.output_sample_rate_hz == 16_000
    assert collector.first_received_at("response.created") is not None
    assert collector.last_received_at("response.audio.delta") is not None


def test_event_collector_reports_engine_token_and_audio_intervals():
    collector = EventCollector()
    collector.add(
        {"type": "response.created", "response": {"id": "resp-a"}},
        received_at_s=10.0,
    )
    stage_metrics = {
        "0": {
            "num_tokens_out": 4,
            "vllm_ttft_ms": 120.0,
            "vllm_tpot_ms": 15.0,
            "vllm_itl_ms": 14.0,
            "vllm_itls_ms": [10.0, 14.0, 18.0],
        }
    }
    for received_at_s, cumulative_audio_ms in ((10.2, 80), (10.25, 160), (10.36, 240)):
        collector.add(
            {
                "type": "response.audio.delta",
                "response_id": "resp-a",
                "delta": base64.b64encode(b"audio").decode("ascii"),
                "sample_rate_hz": 16_000,
                "metadata": {
                    "audio_duration_ms": cumulative_audio_ms,
                    "vllm_omni": {"stage_metrics": stage_metrics},
                },
            },
            received_at_s=received_at_s,
        )
    collector.add(
        {"type": "response.audio_transcript.delta", "response_id": "resp-a", "delta": ""},
        received_at_s=10.1,
    )
    collector.add(
        {"type": "response.audio_transcript.delta", "response_id": "resp-a", "delta": "hello"},
        received_at_s=10.15,
    )
    collector.add(
        {"type": "response.done", "response": {"id": "resp-a"}},
        received_at_s=10.4,
    )

    timing = collector.timing_summary(
        after_s=10.0,
        input_committed_at_s=9.9,
        response_id="resp-a",
    )

    assert timing["stage0_tokens"] == {
        "source": "engine_stage_metrics",
        "output_token_count": 4,
        "ttft_ms": 120.0,
        "tpot_ms": 15.0,
        "itls_ms": [10.0, 14.0, 18.0],
        "inter_token_interval_ms": {
            "count": 3,
            "mean": 14.0,
            "p50": 14.0,
            "p95": 18.0,
            "max": 18.0,
        },
    }
    assert timing["audio_output"] == {
        "source": "client_monotonic_receive",
        "chunk_count": 3,
        "response_created_to_first_audio_ms": 200.0,
        "commit_to_first_audio_ms": 300.0,
        "inter_chunk_interval_ms": {
            "count": 2,
            "mean": 80.0,
            "p50": 50.0,
            "p95": 110.0,
            "max": 110.0,
        },
        "chunk_duration_ms": {
            "count": 3,
            "mean": 80.0,
            "p50": 80.0,
            "p95": 80.0,
            "max": 80.0,
        },
        "max_chunk_gap_ms": 110.0,
    }
    # Raw data only: derived metrics such as the RTF are the caller's job
    # (e.g. vllm_omni.metrics.definitions.compute_audio_rtf).
    assert timing["request_metrics"] == {
        "source": "client_monotonic_receive",
        "measurement_origin": {
            "ttft": "input_audio_buffer.commit client send to first non-empty text delta",
            "ttfp": "input_audio_buffer.commit client send to first audio packet",
        },
        "ttft_ms": 250.0,
        "ttfp_ms": 300.0,
        "audio_generation_ms": 460.0,
        "audio_duration_ms": 240.0,
    }


def test_response_timing_ignores_unowned_session_level_metrics():
    collector = EventCollector()
    collector.add(
        {"type": "response.created", "response": {"id": "resp-a"}},
        received_at_s=10.0,
    )
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(b"audio").decode("ascii"),
            "metadata": {
                "vllm_omni": {
                    "stage_metrics": {
                        "0": {
                            "num_tokens_out": 20,
                            "vllm_ttft_ms": 157.0,
                            "vllm_tpot_ms": 16.0,
                            "vllm_itls_ms": [15.0, 17.0],
                        }
                    }
                }
            },
        },
        received_at_s=10.2,
    )
    collector.add(
        {
            "type": "response.listen",
            "metadata": {
                "vllm_omni": {
                    "stage_metrics": {
                        "0": {
                            "num_tokens_out": 2,
                            "vllm_ttft_ms": 106.0,
                            "vllm_tpot_ms": 0.0,
                            "vllm_itls_ms": [],
                        }
                    }
                }
            },
        },
        received_at_s=10.3,
    )

    timing = collector.timing_summary(after_s=10.0, response_id="resp-a")

    assert timing["stage0_tokens"]["output_token_count"] == 20
    assert timing["stage0_tokens"]["ttft_ms"] == 157.0


def test_summarize_session_request_metrics_averages_audio_turns():
    summary = summarize_session_request_metrics(
        [
            {"ttft_ms": 100.0, "ttfp_ms": 200.0, "rtf": 0.5},
            {"ttft_ms": 300.0, "ttfp_ms": 400.0, "rtf": 0.7},
        ],
        session_id="sess-1",
    )
    assert summary == {
        "session_id": "sess-1",
        "audio_turn_count": 2,
        "mean_ttft_ms": 200.0,
        "mean_ttfp_ms": 300.0,
        "mean_rtf": 0.6,
    }


def test_pcm16_wav_round_trip(tmp_path):
    path = tmp_path / "audio.wav"
    pcm16 = b"\x01\x00\x02\x00"

    write_pcm16_wav(path, pcm16, sample_rate_hz=16_000)

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16_000
    assert read_pcm16_wav(path) == pcm16


# ---------------------------------------------------------------------------
# Model-unit helpers and camera-frame interleaving


def test_duplex_unit_boundary_and_residual_math():
    assert duplex_unit_boundary_ms(0) == 1030
    assert duplex_unit_boundary_ms(2) == 3030
    assert has_residual_model_unit(b"\x00" * 32_000, chunk_period_ms=1000) is False
    assert has_residual_model_unit(b"\x00" * 32_002, chunk_period_ms=1000) is True
    created = {"type": "session.created", "session": {"capabilities": {"chunk_period_ms": 500}}}
    assert chunk_period_ms([created]) == 500
    assert chunk_period_ms([{"type": "session.created", "session": {}}]) == 1000


async def test_stream_pcm_sends_each_units_composite_beside_its_base_frame():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        frames_sent = await client.stream_pcm(
            b"\x01\x00" * (16_000 * 3),
            chunk_ms=200,
            realtime=False,
            video_frames=["f0", "f1"],
            stacked_video_frames=["s0", None],
        )
        appends = [event for event in sock.sent if event.get("type") == "input_audio_buffer.append"]
        # A composite belongs to the unit it was captured in, so it rides the
        # same append as that unit's base frame; a unit without one sends the
        # base alone. Frame k rides the append that closes model unit k.
        assert [event["video_frames"] for event in appends if "video_frames" in event] == [["f0", "s0"], ["f1"]]
        assert [event["audio_end_ms"] for event in appends if "video_frames" in event] == [1200, 2200]
        assert frames_sent == 2
        sock.feed(SESSION_CLOSED)


def test_build_realtime_url_native_duplex_flag_and_http_scheme():
    url = build_realtime_url("http://localhost:8099/v1/realtime", None, native_duplex=None)
    parts = urlsplit(url)
    assert parts.scheme == "ws"
    assert parse_qs(parts.query) == {"duplex": ["1"]}

    url = build_realtime_url("https://host/v1/realtime", "m", native_duplex=False)
    parts = urlsplit(url)
    assert parts.scheme == "wss"
    assert parse_qs(parts.query)["native_duplex"] == ["0"]

    with pytest.raises(ValueError):
        build_realtime_url("ftp://host/v1/realtime", "m")


def test_reference_audio_data_url(tmp_path):
    assert reference_audio_data_url(None) is None
    path = tmp_path / "ref.wav"
    path.write_bytes(b"RIFF")
    assert reference_audio_data_url(str(path)) == "data:audio/wav;base64," + base64.b64encode(b"RIFF").decode("ascii")
    with pytest.raises(FileNotFoundError):
        reference_audio_data_url(str(tmp_path / "missing.wav"))


def test_event_collector_response_text_joins_deltas_per_response():
    collector = EventCollector()
    collector.add({"type": "response.created", "response_id": "r1"})
    collector.add({"type": "response.audio_transcript.delta", "response_id": "r1", "delta": "he"})
    collector.add({"type": "response.output_text.delta", "response_id": "r2", "delta": "other"})
    collector.add({"type": "response.text.delta", "response_id": "r1", "delta": "llo"})
    assert collector.response_text("r1") == "hello"
    assert collector.response_text("r2") == "other"
    assert collector.response_text("r3") == ""
