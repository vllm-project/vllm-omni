# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Playback controls validate their target and release the next-turn gate."""

import copy
from dataclasses import replace

import pytest

from tests.engine.duplex.test_engine_session import _session
from tests.engine.duplex.test_session_runner import append_audio, close_harness, open_harness
from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("target", ["none", "text", "deleted_current", "pending_text", "replaced_current"])
def test_idless_hard_ack_rejects_missing_current_audio_without_mutation(target):
    session = _session()
    if target == "pending_text":
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
    count = 3 if target == "deleted_current" else int(target != "none")
    for _ in range(count):
        response = session.begin_response()
        session.append_assistant_text("hello")
        if target in {"deleted_current", "pending_text", "replaced_current"}:
            session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        session.register_history_item(f"item_{response}", session.end_response())
    if target == "deleted_current":
        assert session.delete_history_item(f"item_{response}")
    elif target == "pending_text":
        response = session.begin_response()
        session.append_assistant_text("text without audio")
        session.end_response(commit_text=False)
        assert len(session.pending_history_item_ids) == 2
    elif target == "replaced_current":
        apply_playback_ack(session, {"response_id": response, "played_ms": 400})
        session.register_history_item(f"item_{response}", {"role": "assistant", "content": "imported replacement"})
        assert not session.history_item_has_audio(f"item_{response}")
    session.acknowledge_playback(200)
    before = copy.deepcopy((session._conversation, session._playback))
    events = apply_playback_ack(session, {"played_ms": 0, "truncate": True})
    assert len(events) == 1 and events[0].type == "error"
    assert events[0].code == "bad_event"
    assert (session._conversation, session._playback) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_history", ["deleted", "uncommitted", "stored"])
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("item_id", ["imported-context", "item_imported"])
async def test_idless_ack_does_not_adopt_imported_history(reply_history, hard, item_id):
    h = await open_harness()
    try:
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="commit_all_on_done"))
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        message = session.end_response(commit_text=reply_history != "uncommitted")
        if message is not None:
            session.register_history_item(f"item_{response}", message)
        if reply_history == "deleted":
            events = await h.run(commands.DeleteItem(item_id=f"item_{response}"))
            assert not any(event.type == "error" for event in events), events
        events = await h.run(
            commands.CreateItem(
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "keep this context"}],
                }
            )
        )
        assert not any(event.type == "error" for event in events), events
        assert session.last_response_id == response
        assert session.active_response_id is None
        assert not session.pending_history_item_ids
        assert session.history_item_ids[item_id]["content"] == "keep this context"
        before = copy.deepcopy((session._conversation, session._playback))
        events = await h.run(
            commands.command_from_realtime({"type": "playback.ack", "played_ms": 200, "truncate": hard})
        )
        assert len(events) == 1, events
        if reply_history == "stored":
            assert events[0].type == "playback.acknowledged"
            assert events[0].details["item_id"] == f"item_{response}"
            assert events[0].details["playback"]["committed_ms"] == 200
            assert session.history_item_ids[f"item_{response}"]["content"] == "he"
        elif hard:
            assert events[0].type == "error" and events[0].code == "bad_event"
            assert (session._conversation, session._playback) == before
        else:
            assert events[0].type == "playback.acknowledged"
            assert events[0].details["item_id"] is None
            assert events[0].details["history_committed"] is False
            assert session.playback.played_ms == session.playback.committed_ms == 200
            assert session._conversation == before[0]
        assert session.history_item_ids[item_id]["content"] == "keep this context"
    finally:
        await close_harness(h)


@pytest.mark.parametrize("hard", [False, True])
def test_explicit_ack_for_imported_non_response_item_is_still_rejected(hard):
    session = _session()
    session.register_history_item("imported-context", {"role": "assistant", "content": "keep this context"})
    before = copy.deepcopy((session._conversation, session._playback))
    events = apply_playback_ack(session, {"item_id": "imported-context", "played_ms": 200, "truncate": hard})
    assert len(events) == 1 and events[0].type == "error"
    assert events[0].code == "playback_item_not_found"
    assert (session._conversation, session._playback) == before


@pytest.mark.parametrize("target", ["active", "pending", "stored", "retired"])
@pytest.mark.parametrize("hard", [False, True])
def test_idless_ack_resolves_one_audio_target_before_mutation(target, hard):
    session = _session()
    if target == "pending":
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
    response = session.begin_response()
    session.append_assistant_text("hello")
    session.mark_audio_sent(400, text_chars=5, audio_complete=True)
    if target != "active":
        session.register_history_item(f"item_{response}", session.end_response())
    if target == "retired":
        apply_playback_ack(session, {"response_id": response, "played_ms": 400})
        assert response not in session._playback.by_response
        assert session.history_item_has_audio(f"item_{response}")
    events = apply_playback_ack(session, {"played_ms": 200 if hard else 400, "truncate": hard})
    assert len(events) == 1 and events[0].type == "playback.acknowledged"
    assert events[0].details["item_id"] == f"item_{response}"
    if target != "active":
        assert session.history_item_ids[f"item_{response}"]["content"] == ("he" if hard else "hello")
        assert response not in session._playback.by_response


@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("older_pending", [False, True])
def test_idless_ack_resolves_retired_audio_only_item(hard, older_pending):
    session = _session()
    if older_pending:
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        session.begin_response()
        session.append_assistant_text("older answer")
        session.mark_audio_sent(600, text_chars=12, audio_complete=True)
        session.end_response()
    pending_before = dict(session.pending_history_item_ids)
    response = session.begin_response()
    session.mark_audio_sent(400, audio_complete=True)
    session.end_response()
    apply_playback_ack(session, {"response_id": response, "played_ms": 400})
    item_id = f"item_{response}"
    assert response not in session._playback.by_response
    assert dict(session.pending_history_item_ids) == pending_before
    assert session.history_item_ids[item_id] == {"role": "assistant", "content": ""}
    assert session.history_item_has_audio(item_id)
    events = apply_playback_ack(session, {"played_ms": 200 if hard else 400, "truncate": hard})
    assert len(events) == 1 and events[0].type == "playback.acknowledged"
    assert events[0].details["item_id"] == item_id
    assert session.history_item_ids[item_id]["content"] == ""
    assert response not in session._playback.by_response


@pytest.mark.asyncio
@pytest.mark.parametrize("item_id", ["imported-context", "item_imported"])
async def test_idless_ack_after_imported_history_allows_next_manual_response(item_id):
    h = await open_qwen(requires_audio=False)
    try:
        await h.run(commands.UpdateSession(patch={"turn_detection": None}))
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        session.end_response()
        events = await h.run(commands.DeleteItem(item_id=f"item_{response}"))
        assert not any(event.type == "error" for event in events), events
        events = await h.run(
            commands.CreateItem(
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "keep this context"}],
                }
            )
        )
        assert not any(event.type == "error" for event in events), events
        await h.run(append_audio())
        await h.run(commands.Commit(final=True, create_response=False))
        events = await h.run(commands.CreateResponse())
        assert any(getattr(event, "code", None) == "response_already_active" for event in events)
        assert not h.port.submissions
        events = await h.run(commands.command_from_realtime({"type": "playback.ack", "played_ms": 400}))
        assert len(events) == 1 and events[0].type == "playback.acknowledged"
        assert events[0].details["item_id"] is None
        assert session.playback.sent_ms == session.playback.committed_ms == 400
        assert session.history_item_ids[item_id]["content"] == "keep this context"
        events = await h.run(commands.CreateResponse())
        assert not any(event.type == "error" for event in events), events
        assert len(h.port.submissions) == 1
        assert session.active_response_id is not None and session.active_response_id != response
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("older_acknowledged", [False, True])
@pytest.mark.parametrize("hard", [False, True])
async def test_idless_ack_does_not_select_older_reply_after_current_item_deleted(older_acknowledged, hard):
    h = await open_qwen(requires_audio=False)
    try:
        await h.run(commands.UpdateSession(patch={"turn_detection": None}))
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        older = session.begin_response()
        session.append_assistant_text("older answer")
        session.mark_audio_sent(600, text_chars=12, audio_complete=True)
        session.end_response()
        if older_acknowledged:
            await h.run(commands.AckPlayback(response_id=older, played_ms=600))
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        session.end_response()
        events = await h.run(commands.DeleteItem(item_id=f"item_{response}"))
        assert not any(event.type == "error" for event in events), events
        assert session.history_item_has_audio(f"item_{older}")
        assert session.last_response_id == response
        before = copy.deepcopy((session._conversation, session._playback))
        events = await h.run(
            commands.command_from_realtime({"type": "playback.ack", "played_ms": 400, "truncate": hard})
        )
        assert len(events) == 1
        if hard:
            assert events[0].type == "error" and events[0].code == "bad_event"
            assert (session._conversation, session._playback) == before
            events = await h.run(commands.AckPlayback(played_ms=400))
        assert events[0].type == "playback.acknowledged"
        assert events[0].details["item_id"] is None
        assert events[0].details["history_committed"] is False
        assert session._conversation == before[0]
        assert session._playback.by_response == before[1].by_response
        assert session.playback.sent_ms == session.playback.committed_ms == 400
        await h.run(append_audio())
        await h.run(commands.Commit(final=True, create_response=False))
        events = await h.run(commands.CreateResponse())
        assert not any(event.type == "error" for event in events), events
        assert len(h.port.submissions) == 1
        assert session.active_response_id is not None and session.active_response_id != response
    finally:
        await close_harness(h)


@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("finished", [False, True])
def test_idless_ack_selects_current_reply_instead_of_older_pending_reply(hard, finished):
    session = _session()
    session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
    older = session.begin_response()
    session.append_assistant_text("older answer")
    session.mark_audio_sent(600, text_chars=12, audio_complete=True)
    session.end_response()
    pending_before = dict(session.pending_history_item_ids)
    response = session.begin_response()
    session.append_assistant_text("hello")
    session.mark_audio_sent(400, text_chars=5, audio_complete=True)
    if finished:
        session.end_response()
        assert len(session.pending_history_item_ids) == 2
    events = apply_playback_ack(session, {"played_ms": 200 if hard else 400, "truncate": hard})
    assert len(events) == 1 and events[0].type == "playback.acknowledged"
    assert events[0].details["item_id"] == f"item_{response}"
    assert events[0].details["playback"]["committed_ms"] == (200 if hard else 400)
    assert session.playback_for_response(older).committed_ms == 0
    assert dict(session.pending_history_item_ids) == pending_before


@pytest.mark.asyncio
@pytest.mark.parametrize("older_acknowledged", [False, True])
@pytest.mark.parametrize("end_before_ack", [False, True])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("hard", [False, True])
async def test_idless_ack_follows_current_reply_through_history_transitions(
    older_acknowledged, end_before_ack, aligned, hard
):
    h = await open_qwen(requires_audio=False)
    try:
        await h.run(commands.UpdateSession(patch={"turn_detection": None}))
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        older = session.begin_response()
        session.append_assistant_text("older answer")
        session.mark_audio_sent(600, text_chars=12, audio_complete=True)
        session.end_response()
        if older_acknowledged:
            await h.run(commands.AckPlayback(response_id=older, played_ms=600))
        older_cursor = session.playback_for_response(older)
        older_item = copy.deepcopy(
            session.history_item_ids.get(f"item_{older}") or session.pending_history_item_ids[f"item_{older}"]
        )
        for imported in ("imported-context", "item_imported"):
            events = await h.run(
                commands.CreateItem(
                    item={
                        "id": imported,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "keep this context"}],
                    }
                )
            )
            assert not any(event.type == "error" for event in events), events
        response = session.begin_response()
        item_id = f"item_{response}"
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        await h.settle()
        session.append_assistant_text("hello")
        session.mark_audio_sent(
            400,
            text_chars=5 if aligned else None,
            text_requires_complete_audio=not aligned,
            audio_complete=True,
        )
        if end_before_ack:
            session.end_response()
            assert item_id in session.pending_history_item_ids
        events = await h.run(commands.command_from_realtime({"type": "playback.ack", "played_ms": 200}))
        assert len(events) == 1 and events[0].type == "playback.acknowledged", events
        assert events[0].details["item_id"] == item_id
        assert session.playback.committed_ms == 200
        if not end_before_ack:
            session.register_history_item(item_id, session.end_response())
        if aligned:
            assert item_id not in session.pending_history_item_ids
            assert session.history_item_ids[item_id]["content"] == "he"
        # A partial ACK must still block the next response, without committing
        # another user input (which would correctly make an unreserved ACK late).
        events = await h.run(commands.CreateResponse())
        assert any(getattr(event, "code", None) == "response_already_active" for event in events)
        assert not h.port.submissions
        events = await h.run(
            commands.command_from_realtime(
                {"type": "playback.ack", "played_ms": 200 if hard else 400, "truncate": hard}
            )
        )
        assert len(events) == 1 and events[0].type == "playback.acknowledged", events
        assert events[0].details["item_id"] == item_id
        expected = ("he" if aligned else "") if hard else "hello"
        assert session.history_item_ids[item_id]["content"] == expected
        assert item_id not in session.pending_history_item_ids
        assert response not in session._playback.by_response
        # A repeated progress ACK after retirement must not expand a hard cap
        # or settle an old response just because it remains pending.
        events = await h.run(commands.command_from_realtime({"type": "playback.ack", "played_ms": 400}))
        assert len(events) == 1 and events[0].type == "playback.acknowledged", events
        assert events[0].details["item_id"] == item_id
        assert events[0].details["history_committed"] is False
        assert session.history_item_ids[item_id]["content"] == expected
        assert session.playback_for_response(older) == older_cursor
        assert (
            session.history_item_ids.get(f"item_{older}") or session.pending_history_item_ids[f"item_{older}"]
        ) == older_item
        for imported in ("imported-context", "item_imported"):
            assert session.history_item_ids[imported]["content"] == "keep this context"
        await h.run(append_audio())
        await h.run(commands.Commit(final=True, create_response=False))
        events = await h.run(commands.CreateResponse())
        assert not any(event.type == "error" for event in events), events
        assert len(h.port.submissions) == 1
        newer = session.active_response_id
        assert newer is not None and newer != response
        session.mark_audio_sent(800)
        current = session.playback
        await h.run(commands.AckPlayback(response_id=response, played_ms=400))
        assert session.active_response_id == newer and session.playback == current
        assert session.history_item_ids[item_id]["content"] == expected
    finally:
        await close_harness(h)


@pytest.mark.parametrize("older_pending", [False, True])
@pytest.mark.parametrize("hard", [False, True])
def test_idless_ack_for_current_pending_reply_preserves_late_history_guard(older_pending, hard):
    session = _session()
    session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
    for _ in range(2 if older_pending else 1):
        session.begin_response()
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        session.end_response()
    session.commit_audio_input(transcript="a later user turn")
    before = copy.deepcopy((session._conversation, session._playback))
    events = apply_playback_ack(session, {"played_ms": 200, "truncate": hard})
    assert len(events) == 1 and events[0].type == "error"
    assert events[0].code == "playback_ack_too_late"
    assert (session._conversation, session._playback) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_ack", ["none", "before_next", "after_next"])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("use_ack", [False, True])
async def test_partial_hard_stop_allows_next_manual_response(delayed_ack, aligned, calibration, use_ack):
    async def calibrate(snapshot):
        return 2

    h = await open_qwen(calibrate=calibrate if calibration else None, requires_audio=False)
    try:
        await h.run(commands.UpdateSession(patch={"turn_detection": None}))
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_chars=5 if aligned else None, audio_complete=True)
        session.register_history_item(f"item_{response}", session.end_response())
        if use_ack:
            stop = commands.command_from_realtime(
                {"type": "playback.ack", "response_id": response, "played_ms": 200, "truncate": True}
            )
        else:
            stop = commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=200)
        events = await h.run(stop)
        assert not any(event.type == "error" for event in events)
        expected = session.history_item_ids[f"item_{response}"]["content"]
        if delayed_ack == "before_next":
            await h.run(commands.AckPlayback(response_id=response, played_ms=400))
        await h.run(append_audio())
        await h.run(commands.Commit(final=True, create_response=False))
        events = await h.run(commands.CreateResponse())
        assert not any(event.type == "error" for event in events), events
        assert len(h.port.submissions) == 1
        newer = session.active_response_id
        assert newer is not None and newer != response
        session.mark_audio_sent(800)
        current = session.playback
        if delayed_ack == "after_next":
            await h.run(commands.AckPlayback(response_id=response, played_ms=400))
        assert session.active_response_id == newer and session.playback == current
        assert session.history_item_ids[f"item_{response}"]["content"] == expected
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_partial_ordinary_ack_keeps_next_manual_response_blocked():
    h = await open_qwen()
    try:
        await h.run(commands.UpdateSession(patch={"turn_detection": None}))
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        response = session.begin_response()
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        session.register_history_item(f"item_{response}", session.end_response())
        await h.run(commands.AckPlayback(response_id=response, played_ms=200))
        await h.run(append_audio())
        events = await h.run(commands.CreateResponse())
        assert any(getattr(event, "code", None) == "response_already_active" for event in events)
        assert not h.port.submissions
    finally:
        await close_harness(h)
