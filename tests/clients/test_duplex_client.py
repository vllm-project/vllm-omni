# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the public duplex clients (fake transports, no server).

``DuplexClient`` runs over a fake websocket; ``InlineDuplexClient`` over a
fake ``DuplexOmni`` whose handle records typed commands and replays typed
events. The demux / handshake tests that only need "feed a server event" are
parameterized over both clients.
"""

from __future__ import annotations

import asyncio
import base64
import json
import wave
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from vllm_omni.clients.duplex import (
    AudioFormat,
    ConnectionResumed,
    DuplexClient,
    DuplexClientBase,
    DuplexConnectionError,
    DuplexProtocolError,
    DuplexSessionClosedError,
    ErrorEvent,
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
from vllm_omni.clients.inline_duplex import InlineDuplexClient
from vllm_omni.engine.duplex import commands as duplex_commands
from vllm_omni.engine.duplex import events as duplex_events
from vllm_omni.engine.duplex.messages import DuplexSessionError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "duplex-0123456789abcdef"
# session.created is not journaled by the server, so it carries no server_event_seq.
SESSION_CREATED = {
    "type": "session.created",
    "session": {"id": SESSION_ID, "model": "test-model"},
    "attachment_generation": 1,
    "resume_token": "tok-1",
}
SESSION_CLOSED = {"type": "session.closed", "session_id": SESSION_ID, "reason": "client_close", "server_event_seq": 99}


# ---------------------------------------------------------------------------
# Fake websocket transport


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
# Fake DuplexOmni for the inline client


class _WireEvent:
    """A typed-event stand-in that renders a fixed wire payload."""

    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw

    @property
    def is_terminal(self) -> bool:
        return self.raw.get("type") in {"session.closed", "session.expired"}

    def to_realtime(self) -> dict[str, object]:
        return dict(self.raw)


class FakeInlineHandle:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.closed = False
        self.close_calls = 0
        self.commands: list[duplex_commands.DuplexCommand] = []
        self.consumer_active = False
        self._outbox: asyncio.Queue = asyncio.Queue()

    def deliver(self, event: Any) -> None:
        self._outbox.put_nowait(event)

    def feed(self, raw: dict[str, object]) -> None:
        self.deliver(_WireEvent(raw))

    async def submit(self, command: duplex_commands.DuplexCommand) -> None:
        if self.closed:
            raise DuplexSessionError("closed", code="session_closed", session_id=self.session_id)
        self.commands.append(command)

    async def events(self):
        self.consumer_active = True
        try:
            while True:
                event = await self._outbox.get()
                yield event
                if event.is_terminal:
                    return
        finally:
            self.consumer_active = False

    async def close(self, *, reason: str = "client_close", timeout: float | None = None) -> None:
        self.close_calls += 1
        if self.closed:
            return
        self.closed = True
        self.deliver(duplex_events.SessionClosed(session_id=self.session_id, reason=reason))


class FakeInlineOmni:
    def __init__(self) -> None:
        self.opened: list[dict[str, object]] = []
        self.handles: list[FakeInlineHandle] = []

    async def open_session(self, config: dict[str, object]) -> FakeInlineHandle:
        self.opened.append(dict(config))
        handle = FakeInlineHandle(SESSION_ID)
        self.handles.append(handle)
        handle.deliver(
            duplex_events.SessionCreated(
                session_id=SESSION_ID,
                session={"id": SESSION_ID, "model": config.get("model")},
            )
        )
        return handle


def make_inline_client(**kwargs) -> tuple[InlineDuplexClient, FakeInlineOmni]:
    omni = FakeInlineOmni()
    kwargs.setdefault("handshake_timeout_s", 5.0)
    return InlineDuplexClient(omni, model="test-model", **kwargs), omni


# ---------------------------------------------------------------------------
# One rig over both clients: ``feed`` a server event, drive ``client``


class Rig:
    def __init__(self, client: DuplexClientBase, feed: Callable[[dict[str, object]], None]) -> None:
        self.client = client
        self.feed = feed


def _websocket_rig(*, handshake: dict[str, object] | None = SESSION_CREATED) -> Rig:
    sock = FakeSocket()
    if handshake is not None:
        sock.feed(handshake)
    client, _ = make_client(sock)
    return Rig(client, sock.feed)


def _inline_rig(*, handshake: dict[str, object] | None = SESSION_CREATED) -> Rig:
    client, omni = make_inline_client()
    original_open = omni.open_session

    async def open_session(config: dict[str, object]) -> FakeInlineHandle:
        handle = await original_open(config)
        if handshake is not SESSION_CREATED:
            # Replace the default session.created with the requested first event.
            handle._outbox = asyncio.Queue()
            if handshake is not None:
                handle.feed(handshake)
        return handle

    omni.open_session = open_session  # type: ignore[method-assign]

    def feed(raw: dict[str, object]) -> None:
        omni.handles[-1].feed(raw)

    return Rig(client, feed)


@pytest.fixture(params=["websocket", "inline"])
def make_rig(request: pytest.FixtureRequest) -> Callable[..., Rig]:
    return _websocket_rig if request.param == "websocket" else _inline_rig


# ---------------------------------------------------------------------------
# SessionConfig


def test_session_config_payload_defaults():
    payload = SessionConfig().to_session_payload(model="m")
    assert payload["model"] == "m"
    assert "session_id" not in payload
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
# Handshake and lifecycle (both clients)


@pytest.mark.asyncio
async def test_handshake_adopts_server_allocated_session_id(make_rig):
    rig = make_rig()
    async with rig.client as client:
        assert client.session_id == SESSION_ID
        assert client.session_info == {"id": SESSION_ID, "model": "test-model"}
        rig.feed(SESSION_CLOSED)


@pytest.mark.asyncio
async def test_handshake_error_raises_protocol_error(make_rig):
    rig = make_rig(handshake={"type": "error", "error": {"code": "unsupported_audio_format", "message": "bad"}})
    with pytest.raises(DuplexProtocolError) as excinfo:
        async with rig.client:
            pass
    assert excinfo.value.code == "unsupported_audio_format"


@pytest.mark.asyncio
async def test_session_expired_raises_from_event_stream(make_rig):
    rig = make_rig()
    async with rig.client as client:
        rig.feed(
            {"type": "session.expired", "session_id": SESSION_ID, "reason": "lease_expired", "server_event_seq": 2}
        )
        with pytest.raises(DuplexSessionClosedError) as excinfo:
            async for _ in client.events():
                pass
        assert "lease_expired" in excinfo.value.reason


# ---------------------------------------------------------------------------
# Websocket transport specifics


@pytest.mark.asyncio
async def test_websocket_handshake_sends_session_update_and_acks_sequenced_events():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, calls = make_client(sock)
    async with client:
        assert client.session_id == SESSION_ID
        assert client.resume_token == "tok-1"
        assert calls == ["ws://test-host:8099/v1/realtime?duplex=1&model=test-model&autostart=0"]
        assert sock.sent[0]["type"] == "session.update"
        assert sock.sent[0]["session"]["model"] == "test-model"
        assert "session_id" not in sock.sent[0]["session"]
        # session.created is unjournaled (no seq): nothing to ack yet.
        assert _acks(sock) == []
        sock.feed({"type": "response.listen", "server_event_seq": 1})
        await _drain(lambda: {"type": "session.event_ack", "server_event_seq": 1} in _acks(sock))
        close_task = asyncio.create_task(client.close())
        await _drain(lambda: "session.close" in sock.sent_types())
        sock.feed(SESSION_CLOSED)
        await close_task
    assert "session.close" in sock.sent_types()


@pytest.mark.asyncio
async def test_resync_required_drops_resume_credential():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, calls = make_client(sock, reconnect=ReconnectPolicy(max_attempts=2, backoff_s=(0.0, 0.0)))
    async with client:
        assert client.resume_token == "tok-1"
        stream = client.events()
        sock.feed({"type": "session.resync_required", "session_id": SESSION_ID, "server_event_seq": 2})
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
# Input events (websocket wire shape)


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_stream_pcm_chunking():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        await client.stream_pcm(b"\x00" * 8000, chunk_ms=100, realtime=False)
        appends = [event for event in sock.sent if event.get("type") == "input_audio_buffer.append"]
        assert [a["duration_ms"] for a in appends] == [100, 100, 50]
        sock.feed(SESSION_CLOSED)


@pytest.mark.asyncio
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
# Response demultiplexing (both clients)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.mark.asyncio
async def test_response_handle_flow(make_rig):
    rig = make_rig()
    chunk = b"\x00\x01" * 2400  # 100 ms of pcm16 @ 24 kHz
    async with rig.client as client:
        rig.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        rig.feed({"type": "response.speak", "response_id": "resp-1", "server_event_seq": 3})
        rig.feed(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp-1",
                "delta": _b64(chunk),
                "sample_rate_hz": 24_000,
                "server_event_seq": 4,
            }
        )
        rig.feed(
            {
                "type": "response.output_audio_transcript.delta",
                "response_id": "resp-1",
                "delta": "hi there",
                "server_event_seq": 5,
            }
        )
        rig.feed({"type": "response.done", "response_id": "resp-1", "server_event_seq": 6})

        async for response in client.responses():
            chunks = [piece async for piece in response.audio()]
            assert chunks == [chunk]
            assert response.decision == "speak"
            assert response.transcript == "hi there"
            assert response.played_ms == pytest.approx(100.0)
            assert response.finished
            break
        rig.feed(SESSION_CLOSED)


@pytest.mark.asyncio
async def test_listen_decision_yields_finished_silent_handle(make_rig):
    rig = make_rig()
    async with rig.client as client:
        rig.feed({"type": "response.listen", "server_event_seq": 2})
        async for response in client.responses():
            assert response.decision == "listen"
            assert response.finished
            assert [piece async for piece in response.audio()] == []
            break
        rig.feed(SESSION_CLOSED)


@pytest.mark.asyncio
async def test_listen_terminates_active_response(make_rig):
    rig = make_rig()
    async with rig.client as client:
        rig.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        rig.feed({"type": "response.listen", "response_id": "resp-1", "server_event_seq": 3})
        async for response in client.responses():
            await response.wait(timeout_s=5.0)
            assert response.decision == "listen"
            break
        rig.feed(SESSION_CLOSED)


@pytest.mark.asyncio
async def test_id_less_listen_never_closes_an_in_flight_response(make_rig):
    # An id-less listen is a standalone decision beat (e.g. a silence-skip
    # while a response streams); it must surface as its own finished handle
    # and leave the in-flight response to its real terminal.
    rig = make_rig()
    async with rig.client as client:
        rig.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        rig.feed({"type": "response.speak", "response_id": "resp-1", "server_event_seq": 3})
        rig.feed({"type": "response.listen", "server_event_seq": 4})
        rig.feed({"type": "response.done", "response_id": "resp-1", "server_event_seq": 5})
        seen: list[tuple[str | None, str | None, bool]] = []
        async for response in client.responses():
            await response.wait(timeout_s=5.0)
            seen.append((response.response_id, response.decision, response.finished))
            if len(seen) == 2:
                break
        assert ("resp-1", "speak", True) in seen
        assert (None, "listen", True) in seen
        rig.feed(SESSION_CLOSED)


@pytest.mark.asyncio
async def test_terminal_listen_keeps_spoken_decision(make_rig):
    # A terminal listen after the response spoke is the model yielding the
    # turn; the handle must stay decision="speak" with its audio intact.
    rig = make_rig()
    chunk = b"\x00\x01" * 2400
    async with rig.client as client:
        rig.feed({"type": "response.created", "response": {"id": "resp-1"}, "server_event_seq": 2})
        rig.feed({"type": "response.speak", "response_id": "resp-1", "server_event_seq": 3})
        rig.feed(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp-1",
                "delta": _b64(chunk),
                "sample_rate_hz": 24_000,
                "server_event_seq": 4,
            }
        )
        rig.feed({"type": "response.listen", "response_id": "resp-1", "server_event_seq": 5})
        async for response in client.responses():
            chunks = [piece async for piece in response.audio()]
            assert chunks == [chunk]
            assert response.decision == "speak"
            assert response.finished
            break
        rig.feed(SESSION_CLOSED)


@pytest.mark.asyncio
async def test_error_event_surfaces_on_response_path(make_rig):
    # A rejected send produces no response; a consumer waiting in responses()
    # must see the rejection instead of waiting forever.
    rig = make_rig()
    async with rig.client as client:
        rig.feed(
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
        rig.feed(SESSION_CLOSED)


@pytest.mark.asyncio
async def test_slow_audio_consumer_drops_oldest_instead_of_stalling():
    from vllm_omni.clients.duplex import AudioDelta, ResponseHandle

    handle = ResponseHandle(
        response_id="r1",
        output_format=AudioFormat("pcm16", 24_000),
        max_buffered_events=2,
    )
    for payload in (b"c1", b"c2", b"c3"):
        # _feed must never block the reader, even against a full queue.
        handle._feed(AudioDelta({"type": "response.output_audio.delta", "delta": _b64(payload)}))
    handle._finish(None)
    chunks = [chunk async for chunk in handle.audio()]
    assert chunks == [b"c3"]  # oldest chunks were dropped, the sentinel landed


# ---------------------------------------------------------------------------
# Resume (websocket only)


@pytest.mark.asyncio
async def test_resume_after_transport_drop():
    first = FakeSocket()
    first.feed(SESSION_CREATED)
    second = FakeSocket()
    # Mirror the real resume activation payload
    # (entrypoints/duplex/serving.py, activation_payload_factory): it carries
    # the identity fields plus the (empty) public session object.
    second.feed(
        {
            "type": "session.resumed",
            "session_id": SESSION_ID,
            "session": {},
            "attachment_generation": 2,
            "resume_token": "tok-2",
        }
    )
    client, calls = make_client(
        first,
        second,
        reconnect=ReconnectPolicy(max_attempts=2, backoff_s=(0.0, 0.0)),
    )
    async with client:
        stream = client.events()

        async def collect_until_resumed():
            seen = []
            async for event in stream:
                seen.append(event)
                if isinstance(event, SessionResumed):
                    return seen

        consumer = asyncio.create_task(collect_until_resumed())
        await asyncio.sleep(0)  # let the consumer subscribe before the drop
        await asyncio.sleep(0)
        first.feed({"type": "response.listen", "server_event_seq": 7})
        await _drain(lambda: {"type": "session.event_ack", "server_event_seq": 7} in _acks(first))
        first.feed(RuntimeError("transport dropped"))
        seen = await asyncio.wait_for(consumer, timeout=5.0)
        assert [type(event) for event in seen[-2:]] == [ConnectionResumed, SessionResumed]
        assert client.resume_token == "tok-2"
        # The superseded transport is closed, not leaked.
        assert first.closed
        # The activation payload has no session object; the info captured at
        # the original handshake must survive the resume.
        assert client.session_info == {"id": SESSION_ID, "model": "test-model"}
        assert len(calls) == 2
        resume = second.sent[0]
        assert resume == {
            "type": "session.resume",
            "session_id": SESSION_ID,
            "resume_token": "tok-1",
            "last_received_server_event_seq": 7,
        }
        second.feed(SESSION_CLOSED)


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
# Inline client over a fake DuplexOmni


@pytest.mark.asyncio
async def test_inline_handshake_opens_session_with_payload_and_adopts_id():
    client, omni = make_inline_client(config=SessionConfig(ref_audio="data:audio/wav;base64,AAA="))
    async with client:
        assert omni.opened == [client.config.to_session_payload(model="test-model")]
        assert omni.opened[0]["ref_audio"] == "data:audio/wav;base64,AAA="
        assert client.session_id == SESSION_ID
        handle = omni.handles[0]
        # The session was opened with the full session object already; the
        # base handshake's session.update must not be re-submitted as a patch
        # (a ref_audio patch would be rejected by the runner).
        assert handle.commands == []
        await client.send({"type": "session.update", "session": {"instructions": "later"}})
        assert [type(command) for command in handle.commands] == [duplex_commands.UpdateSession]
        assert handle.commands[0].patch == {"instructions": "later"}


@pytest.mark.asyncio
async def test_inline_client_submits_typed_commands_in_order():
    client, omni = make_inline_client()
    async with client:
        handle = omni.handles[0]
        pcm = b"\x01\x00" * 1600
        await client.append_audio(pcm, is_speech=True)
        await client.commit(create_response=True)
        await client.cancel_response("resp-1")
        await client.ack_playback(120.9, response_id="resp-1")
        assert [type(command) for command in handle.commands] == [
            duplex_commands.AppendAudio,
            duplex_commands.Commit,
            duplex_commands.CancelResponse,
            duplex_commands.AckPlayback,
        ]
        append, commit, cancel, ack = handle.commands
        assert append.is_speech is True
        assert append.audio_end_ms == 100
        assert append.event_id
        assert commit.final is True and commit.create_response is True
        assert cancel.response_id == "resp-1"
        assert ack.played_ms == 120 and ack.response_id == "resp-1"


@pytest.mark.asyncio
async def test_inline_session_close_maps_to_handle_close():
    client, omni = make_inline_client()
    async with client:
        pass
    handle = omni.handles[0]
    assert handle.closed
    assert handle.close_calls >= 1
    assert not any(isinstance(command, duplex_commands.CloseSession) for command in handle.commands)
    # The typed session.closed reached the client as the expected end.
    with pytest.raises(DuplexSessionClosedError):
        await client.send({"type": "session.heartbeat"})


@pytest.mark.asyncio
async def test_inline_client_forwards_heartbeats_and_drops_event_acks():
    client, omni = make_inline_client()
    async with client:
        handle = omni.handles[0]
        await client.send({"type": "session.heartbeat"})
        await client.send({"type": "session.event_ack", "server_event_seq": 3})
        # A heartbeat is a lease touch the engine must see; the ack is a journal concern.
        assert [type(command).__name__ for command in handle.commands] == ["Heartbeat"]


@pytest.mark.asyncio
async def test_inline_malformed_payload_surfaces_as_error_event():
    client, omni = make_inline_client()
    async with client:
        handle = omni.handles[0]
        # Subscribe before sending: the translation error is dispatched inline.
        first_event = asyncio.create_task(client.wait_for("error", timeout_s=5.0))
        await asyncio.sleep(0)
        event_id = await client.send({"type": "playback.ack"})  # played_ms missing
        event = await first_event
        assert isinstance(event, ErrorEvent)
        assert event.code == "bad_event"
        assert event.related_event_id == event_id
        assert handle.commands == []
        with pytest.raises(DuplexProtocolError) as excinfo:
            async for _ in client.responses():
                pass
        assert excinfo.value.code == "bad_event"


@pytest.mark.asyncio
async def test_inline_teardown_cancels_pump_and_closes_handle():
    client, omni = make_inline_client()
    async with client:
        handle = omni.handles[0]
        await _drain(lambda: handle.consumer_active)
        pump = client._pump_task
        assert pump is not None and not pump.done()
    assert pump.done()
    assert not handle.consumer_active
    assert handle.closed


@pytest.mark.asyncio
async def test_inline_send_after_handle_closed_raises_connection_error():
    client, omni = make_inline_client()
    async with client:
        handle = omni.handles[0]
        handle.closed = True  # the engine ended the session without an event yet
        with pytest.raises(DuplexConnectionError, match="not open"):
            await client.send({"type": "input_audio_buffer.clear"})
        handle.closed = False


# ---------------------------------------------------------------------------
# Collector


def test_event_collector_accumulates_audio():
    collector = EventCollector()
    collector.add({"type": "response.created", "response": {"id": "r1"}}, received_at_s=1.0)
    collector.add(
        {"type": "response.output_audio.delta", "response_id": "r1", "delta": _b64(b"ab"), "sample_rate_hz": 16_000},
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
        extra_query={"probe": "1"},
    )

    query = parse_qs(urlsplit(url).query)
    assert query == {
        "custom": ["1"],
        "duplex": ["1"],
        "model": ["openbmb/MiniCPM-o-4_5"],
        "probe": ["1"],
    }


def test_build_realtime_url_resume_only_when_autostart_disabled():
    url = build_realtime_url(
        "ws://localhost:8099/v1/realtime?duplex=1",
        "openbmb/MiniCPM-o-4_5",
        autostart=False,
    )

    query = parse_qs(urlsplit(url).query)
    assert query["autostart"] == ["0"]
    assert "session_id" not in query


def test_event_collector_partitions_audio_by_response():
    collector = EventCollector()
    collector.add({"type": "response.created", "response": {"id": "resp-a"}})
    collector.add(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(b"audio-a").decode("ascii"),
            "sample_rate_hz": 16_000,
        }
    )

    assert collector.response_ids == ["resp-a"]
    assert collector.audio_bytes("resp-a") == b"audio-a"
    assert collector.output_sample_rate_hz == 16_000
    assert collector.first_received_at("response.created") is not None
    assert collector.last_received_at("response.output_audio.delta") is not None


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
                "type": "response.output_audio.delta",
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
        {"type": "response.output_audio_transcript.delta", "response_id": "resp-a", "delta": ""},
        received_at_s=10.1,
    )
    collector.add(
        {"type": "response.output_audio_transcript.delta", "response_id": "resp-a", "delta": "hello"},
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
            "type": "response.output_audio.delta",
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


@pytest.mark.asyncio
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


def test_build_realtime_url_rewrites_http_scheme_and_rejects_others():
    url = build_realtime_url("http://localhost:8099/v1/realtime", None)
    parts = urlsplit(url)
    assert parts.scheme == "ws"
    assert parse_qs(parts.query) == {"duplex": ["1"]}

    url = build_realtime_url("https://host/v1/realtime", "m", autostart=True)
    parts = urlsplit(url)
    assert parts.scheme == "wss"
    assert parse_qs(parts.query) == {"duplex": ["1"], "model": ["m"], "autostart": ["1"]}

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
    collector.add({"type": "response.output_audio_transcript.delta", "response_id": "r1", "delta": "he"})
    collector.add({"type": "response.output_text.delta", "response_id": "r2", "delta": "other"})
    collector.add({"type": "response.text.delta", "response_id": "r1", "delta": "llo"})
    assert collector.response_text("r1") == "hello"
    assert collector.response_text("r2") == "other"
    assert collector.response_text("r3") == ""
