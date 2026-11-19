# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Audio truncation cannot mutate context that has never produced audio."""

import copy

import pytest

from tests.engine.duplex.test_engine_session import _session
from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack
from vllm_omni.protocol.realtime.items import validate_realtime_item_truncate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text", "output_audio"])
@pytest.mark.parametrize("cutoff", [0, 200])
async def test_imported_context_without_audio_rejects_truncate_without_side_effects(content_type, cutoff):
    h = await open_harness()
    try:
        item_id = "item_imported"
        part = {"type": content_type, "text" if content_type == "text" else "transcript": "keep this context"}
        content = [part]
        if content_type == "output_audio":
            content.append({"type": "text", "text": "keep this context"})
        await h.run(
            commands.CreateItem(item={"id": item_id, "type": "message", "role": "assistant", "content": content})
        )
        assert h.session.history_item_ids[item_id]["content"] == "keep this context"
        projector = h.runner.out.projector
        before = copy.deepcopy((h.session.history, projector.conversation_items, projector.item_truncation_cursors))
        events = await h.run(commands.TruncateItem(item_id=item_id, audio_end_ms=cutoff, event_id="no-audio"))
        assert len(events) == 1 and events[0].type == "error", events
        assert events[0].code == "bad_event"
        assert events[0].related_event_id == "no-audio"
        assert (h.session.history, projector.conversation_items, projector.item_truncation_cursors) == before
        assert not h.session._conversation.hard_truncations_ms
        assert not h.session._playback.by_response
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("ended", [False, True])
@pytest.mark.parametrize("native_signal", [False, True])
async def test_text_only_response_truncation_preserves_text_and_future_audio(ended, native_signal):
    h = await open_harness()
    try:
        session = h.session
        response = session.begin_response()
        item_id = f"item_{response}"
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        if ended:
            session.register_history_item(item_id, session.end_response())
            session.finalize_response_playback(response)
        await h.settle()
        before = copy.deepcopy(
            (session.history, session._playback.by_response, session._conversation.hard_truncations_ms)
        )
        if native_signal:
            command = commands.SignalTurn(
                event="conversation.item.truncate", signal_payload={"item_id": item_id, "audio_end_ms": 0}
            )
        else:
            command = commands.TruncateItem(item_id=item_id, audio_end_ms=0)
        events = await h.run(command)
        assert len(events) == 1 and events[0].type == "error", events
        assert events[0].code == "bad_event"
        assert (session.history, session._playback.by_response, session._conversation.hard_truncations_ms) == before
        assert item_id not in h.runner.out.projector.item_truncation_cursors
        if not ended:
            # Rejecting a premature truncate must not cap subsequently generated audio.
            session.mark_audio_sent(400, text_chars=5, audio_complete=True)
            session.register_history_item(item_id, session.end_response())
            await h.run(commands.AckPlayback(response_id=response, played_ms=400))
        assert session.history_item_ids[item_id]["content"] == "hello"
    finally:
        await close_harness(h)


@pytest.mark.parametrize("ended", [False, True])
def test_internal_truncate_without_audio_is_a_noop(ended):
    session = _session()
    response = session.begin_response()
    session.append_assistant_text("hello")
    item_id = f"item_{response}"
    if ended:
        session.register_history_item(item_id, session.end_response())
        session.finalize_response_playback(response)
    assert session.truncate_history_item(item_id, audio_end_ms=200, hard=True) is False
    assert not session._conversation.hard_truncations_ms
    assert not session._conversation.pending_truncations_ms
    if ended:
        assert session.history_item_ids[item_id]["content"] == "hello"


@pytest.mark.parametrize("duration_key", ["audio_duration_ms", "duration_ms", "audio_ms"])
def test_zero_duration_is_a_known_audio_bound(duration_key):
    item = {"type": "message", "role": "assistant", "content": [{"type": "output_audio", duration_key: 0}]}
    assert validate_realtime_item_truncate(item, content_index=0, audio_end_ms=1) == (
        "conversation.item.truncate audio_end_ms exceeds item audio duration"
    )


def test_no_audio_hard_ack_rejection_does_not_reserve_a_history_slot():
    session = _session()
    response = session.begin_response()
    session.append_assistant_text("hello")
    before = copy.deepcopy((session.history, session._playback.by_response))
    events = apply_playback_ack(session, {"response_id": response, "played_ms": 200, "truncate": True})
    assert len(events) == 1 and events[0].type == "error"
    assert events[0].code == "bad_event"
    assert (session.history, session._playback.by_response) == before
    assert not session._conversation.history_item_placeholders
    assert not session._conversation.hard_truncations_ms


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("delete_first", [False, True])
@pytest.mark.parametrize("hard_stopped", [False, True])
def test_replacing_retired_history_does_not_inherit_its_audio(aligned, delete_first, hard_stopped):
    session = _session()
    response = session.begin_response()
    item_id = f"item_{response}"
    session.append_assistant_text("old answer")
    session.mark_audio_sent(400, text_chars=10 if aligned else None, audio_complete=True)
    session.register_history_item(item_id, session.end_response())
    apply_playback_ack(
        session, {"response_id": response, "played_ms": 200 if hard_stopped else 400, "truncate": hard_stopped}
    )
    assert response not in session._playback.by_response
    assert session.history_item_has_audio(item_id)
    if delete_first:
        assert session.delete_history_item(item_id)
    session.register_history_item(item_id, {"role": "assistant", "content": "replacement context"})
    assert session.truncate_history_item(item_id, audio_end_ms=0, hard=True) is False
    assert session.history_item_ids[item_id]["content"] == "replacement context"
    assert item_id not in session._conversation.retired_audio_item_ids
    assert item_id not in session._conversation.item_audio_text_marks
    assert item_id not in session._conversation.heard_prefixes
    assert item_id not in session._conversation.hard_truncations_ms
    apply_playback_ack(session, {"response_id": response, "played_ms": 400})
    assert session.history_item_ids[item_id]["content"] == "replacement context"
