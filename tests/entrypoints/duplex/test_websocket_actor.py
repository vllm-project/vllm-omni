# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest

from vllm_omni.entrypoints.duplex.websocket import (
    DuplexWebSocketActor,
    normalize_duplex_input_event,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(dict(payload))


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"type": "signal_turn", "event": "barge_in"}, {"type": "turn.signal", "event": "barge_in"}),
        ({"type": "close_session"}, {"type": "session.close"}),
        ({"type": "audio.playback_ack", "played_ms": 1}, {"type": "playback.ack", "played_ms": 1}),
        ({"type": "input_text.append", "text": "a"}, {"type": "input.text.append", "text": "a"}),
        ({"type": "push_text", "text": "b"}, {"type": "input.text.append", "text": "b"}),
        (
            {"type": "input.audio.append", "audio": "wav"},
            {"type": "input_audio_buffer.append", "audio": "wav", "format": "wav"},
        ),
        (
            {"type": "push_chunk", "audio": "wav", "format": "pcm_f32le"},
            {"type": "input_audio_buffer.append", "audio": "wav", "format": "pcm_f32le"},
        ),
    ],
)
def test_input_aliases_normalize_once_at_mailbox_boundary(event, expected):
    assert normalize_duplex_input_event(event) == expected


def test_actor_uses_one_fifo_mailbox_for_inbound_events():
    actor = DuplexWebSocketActor(FakeWebSocket())

    assert isinstance(actor.mailbox, asyncio.Queue)
    assert not isinstance(actor.mailbox, asyncio.PriorityQueue)
    assert not hasattr(actor, "control_queue")
    assert not hasattr(actor, "input_queue")
    assert not hasattr(actor, "event_queue")
    assert not hasattr(actor, "lifecycle_state")
    assert not hasattr(actor, "last_response_id")
    assert not hasattr(actor, "overlap_speech_ms")
    assert not hasattr(actor, "transition")
    assert not hasattr(actor, "output_generation_in_flight")
    assert not hasattr(actor, "session")
    assert not hasattr(actor, "runtime_opened")
    assert not hasattr(actor, "runtime_closed")
    assert not hasattr(actor, "drain_input_queue")
    assert not hasattr(actor, "control_events_seen")
    assert not hasattr(actor, "input_events_seen")
    assert not hasattr(actor, "cancel_count")


@pytest.mark.asyncio
async def test_control_event_does_not_overtake_earlier_audio_input():
    actor = DuplexWebSocketActor(FakeWebSocket())
    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "pcm"})
    await actor.enqueue_event({"type": "response.cancel"})

    assert await actor.next_event() == {"type": "input_audio_buffer.append", "audio": "pcm"}
    assert await actor.next_event() == {"type": "response.cancel"}


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["input.cancel", "session.close", "close_session"])
async def test_terminal_control_preserves_wire_order(event_type: str):
    actor = DuplexWebSocketActor(FakeWebSocket())
    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "pcm"})
    await actor.enqueue_event({"type": event_type})

    assert await actor.next_event() == {"type": "input_audio_buffer.append", "audio": "pcm"}
    assert await actor.next_event() == {"type": event_type}


@pytest.mark.asyncio
async def test_mailbox_delivers_every_enqueued_event_exactly_once():
    actor = DuplexWebSocketActor(FakeWebSocket())
    events = [
        {"type": "input_audio_buffer.append", "audio": str(index)}
        if index % 2
        else {"type": "response.created", "response_id": f"resp-{index}"}
        for index in range(20)
    ]

    await asyncio.gather(*(actor.enqueue_event(dict(event)) for event in events))
    received = [await actor.next_event() for _ in events]

    assert sorted(received, key=repr) == sorted(events, key=repr)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["events", "bytes"])
async def test_mailbox_limits_apply_before_input_reaches_runtime(limit):
    actor = DuplexWebSocketActor(FakeWebSocket())
    actor.max_mailbox_events = 1 if limit == "events" else 256
    actor.max_mailbox_bytes = 80 if limit == "bytes" else 16 * 1024 * 1024
    first = {"type": "input_audio_buffer.append", "audio": "a"}
    await actor.enqueue_event(first)
    with pytest.raises(BufferError, match="mailbox"):
        await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "b" * 80})
    assert actor.mailbox.qsize() == 1
    assert await actor.next_event() == first
    assert not actor.has_queued_input_events()
    await actor.enqueue_event(first)
    assert await actor.next_event() == first


@pytest.mark.asyncio
async def test_full_mailbox_reserves_one_fifo_reader_terminal_and_releases_budget():
    actor = DuplexWebSocketActor(FakeWebSocket(), max_mailbox_events=1)
    event = {"type": "input_audio_buffer.append", "audio": "a"}
    await actor.enqueue_event(event, encoded_bytes=100)
    await actor.enqueue_terminal("__disconnect__")
    await actor.enqueue_terminal("__disconnect__")
    assert actor.mailbox.qsize() == 2
    assert await actor.next_event() == event
    assert await actor.next_event() == {"type": "__disconnect__"}
    assert actor._queued_mailbox_bytes == 0
    assert not actor.has_queued_input_events()


@pytest.mark.asyncio
async def test_overflow_cleanup_discards_only_transport_backlog():
    actor = DuplexWebSocketActor(FakeWebSocket())
    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "a"}, encoded_bytes=100)
    actor.discard_pending_events()
    await asyncio.wait_for(actor.mailbox.join(), timeout=1)
    assert not actor.has_queued_input_events()
    assert actor._queued_mailbox_bytes == 0
    await actor.enqueue_terminal("__mailbox_overflow__")
    assert await actor.next_event() == {"type": "__mailbox_overflow__"}


@pytest.mark.asyncio
async def test_writer_is_single_owner_of_websocket_send():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket)
    writer = asyncio.create_task(actor.writer_loop())

    await actor.send_json({"type": "one"})
    await actor.send_json({"type": "two"})
    await actor.close_writer()
    await writer

    assert websocket.sent == [{"type": "one"}, {"type": "two"}]


@pytest.mark.asyncio
async def test_stale_fence_payload_is_dropped_before_websocket_send():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket, current_epoch=lambda: 2)
    writer = asyncio.create_task(actor.writer_loop())

    await actor.send_json({"type": "response.audio.delta", "epoch": 1})
    await actor.send_json({"type": "response.audio.delta", "epoch": 2})
    await actor.close_writer()
    await writer

    assert websocket.sent == [{"type": "response.audio.delta", "epoch": 2}]
    assert actor.stale_output_dropped == 1


@pytest.mark.asyncio
async def test_writer_does_not_revoke_accepted_terminal_after_close_starts():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket, current_epoch=lambda: 1)
    await actor.send_json({"type": "response.done", "epoch": 1, "response_id": "resp-1"})
    actor.closing = True
    await actor.close_writer()

    await actor.writer_loop()

    assert websocket.sent == [{"type": "response.done", "epoch": 1, "response_id": "resp-1"}]
    assert actor.stale_output_dropped == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["events", "bytes"])
async def test_output_backlog_rejects_overflow_and_close_never_waits(limit):
    actor = DuplexWebSocketActor(
        FakeWebSocket(),
        max_output_events=1 if limit == "events" else 256,
        max_output_bytes=100 if limit == "bytes" else 1024,
    )
    await actor.send_json({"type": "audio", "audio": "a"})
    with pytest.raises(BufferError, match="output"):
        await actor.send_json({"type": "audio", "audio": "b" * 100})
    await asyncio.wait_for(actor.close_writer(), 1)
    await asyncio.wait_for(actor.writer_loop(), 1)
    assert actor.output_queue.empty()
    assert actor._queued_output_bytes == 0
    assert len(actor.websocket.sent) == 1
