# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Playback controls validate their target and release the next-turn gate."""

import copy
from dataclasses import replace

import pytest

from tests.engine.duplex.test_engine_session import _session
from tests.engine.duplex.test_session_runner import append_audio, close_harness
from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("target", ["none", "text", "ambiguous"])
def test_idless_hard_ack_rejects_missing_audio_or_ambiguous_target_without_mutation(target):
    session = _session()
    for _ in range(2 if target == "ambiguous" else int(target == "text")):
        response = session.begin_response()
        session.append_assistant_text("hello")
        if target == "ambiguous":
            session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        session.register_history_item(f"item_{response}", session.end_response())
    session.acknowledge_playback(200)
    before = copy.deepcopy((session._conversation, session._playback))
    events = apply_playback_ack(session, {"played_ms": 0, "truncate": True})
    assert len(events) == 1 and events[0].type == "error"
    assert events[0].code == "bad_event"
    assert (session._conversation, session._playback) == before


@pytest.mark.parametrize("target", ["active", "pending", "stored"])
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
    events = apply_playback_ack(session, {"played_ms": 200 if hard else 400, "truncate": hard})
    assert len(events) == 1 and events[0].type == "playback.acknowledged"
    assert events[0].details["item_id"] == f"item_{response}"
    if target != "active":
        assert session.history_item_ids[f"item_{response}"]["content"] == ("he" if hard else "hello")
        assert response not in session._playback.by_response


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_ack", ["none", "before_next", "after_next"])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("calibration", [False, True])
async def test_partial_hard_stop_allows_next_manual_response(delayed_ack, aligned, calibration):
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
        events = await h.run(commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=200))
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
