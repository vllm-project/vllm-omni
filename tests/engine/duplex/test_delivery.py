# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
from dataclasses import replace

import pytest

from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer, DuplexOutputOverflowError
from vllm_omni.engine.duplex.events import AudioDelta, ResponseDone, SessionClosed, TranscriptDelta
from vllm_omni.engine.duplex.realtime_events import RealtimeProjectionState, project_internal_event

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "delta_field", "content_field"),
    [
        ("response.text.delta", "delta", "text"),
        ("response.output_audio.delta", "text", "transcript"),
    ],
)
async def test_queued_response_creation_keeps_payload_and_byte_count(event_type, delta_field, content_field):
    state = RealtimeProjectionState(session_id="s")
    events = project_internal_event(state, {"type": "response.created", "response_id": "r", "modalities": ["text"]})
    payloads = [event.to_realtime() for event in events]
    byte_count = sum(
        len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) for payload in payloads
    )
    output = DuplexOutputBuffer(max_bytes=byte_count, max_events=len(events))
    for event in events:
        output.put(event)
    assert output.pending_bytes == byte_count

    text = "Growing response. " * 512
    project_internal_event(state, {"type": event_type, "response_id": "r", delta_field: text})
    item = next(iter(state.conversation_items.values()))
    assert item["content"][0][content_field] == text
    assert output.pending_bytes == byte_count
    delivered = [await output.get() for _ in events]
    assert [event.to_realtime() for event in delivered] == payloads
    assert output.pending_events == output.pending_bytes == 0


@pytest.mark.asyncio
async def test_cancel_removes_only_matching_audio_and_invalidates_held_audio():
    output = DuplexOutputBuffer(max_bytes=4096, max_events=8)
    held = AudioDelta(response_id="old", epoch=0, delta="AAAA")
    text = TranscriptDelta(response_id="old", epoch=0, delta="kept")
    other = replace(held, response_id="other")
    newer = replace(held, epoch=1)
    done = ResponseDone(response_id="old", response={"status": "cancelled"})
    for event in (held, replace(held), text, other, newer, done):
        output.put(event)
    assert await output.get() is held
    assert output.is_valid(held)
    output.invalidate("old", through_epoch=0)
    assert output.pending_events == 4
    with output.guard(held) as valid:
        assert not valid
    assert [await output.get() for _ in range(4)] == [text, other, newer, done]
    assert output.pending_events == output.pending_bytes == 0


@pytest.mark.asyncio
async def test_full_reserve_cannot_block_closure_or_reorder_valid_audio():
    output = DuplexOutputBuffer(max_bytes=4096, max_events=1, reserve_events=1)
    audio = AudioDelta(response_id="r", epoch=0, delta="AAAA")
    done = ResponseDone(response_id="r", response={"status": "completed"})
    output.put(audio)
    output.put(done)
    with pytest.raises(DuplexOutputOverflowError):
        output.put(done)
    closed = SessionClosed(session_id="s", reason="client_close")
    output.close(closed)
    output.close(SessionClosed(reason="duplicate"))
    assert not output.put(audio)
    assert [await output.get() for _ in range(4)] == [audio, done, closed, None]
    assert output.pending_events == output.pending_bytes == 0


@pytest.mark.asyncio
async def test_oversized_close_still_ends_stream_without_reopening_it():
    output = DuplexOutputBuffer(max_bytes=128, max_events=1)
    closed = SessionClosed(session_id="s", details={"large": "x" * 5000})
    output.close(closed)
    terminal = await output.get()
    assert isinstance(terminal, SessionClosed)
    assert terminal.event_id == closed.event_id
    assert terminal.session_id == "s"
    assert terminal.reason == "close_details_exceed_output_limit"
    assert terminal.details == {}
    assert await output.get() is None
    output.close(closed)
    assert await output.get() is None


def test_byte_limit_rejects_audio_without_changing_pending_budget():
    output = DuplexOutputBuffer(max_bytes=128, max_events=8)
    with pytest.raises(DuplexOutputOverflowError):
        output.put(AudioDelta(delta="AAAA" * 64))
    assert output.pending_events == output.pending_bytes == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_can_be_replaced_and_woken_from_another_thread():
    output = DuplexOutputBuffer(max_bytes=4096, max_events=8)
    abandoned = asyncio.create_task(output.get())
    await asyncio.sleep(0)
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    waiting = asyncio.create_task(output.get())
    await asyncio.sleep(0)
    event = AudioDelta(response_id="r", epoch=0, delta="AAAA")
    await asyncio.to_thread(output.put, event)
    assert await asyncio.wait_for(waiting, timeout=2) is event
    closing = asyncio.create_task(output.get())
    await asyncio.sleep(0)
    await asyncio.to_thread(output.close)
    assert await asyncio.wait_for(closing, timeout=2) is None
