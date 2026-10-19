# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A response finishes only after output and playback have both completed."""

import pytest

from tests.engine.duplex.test_engine_session import _session
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_CONTENT = [
    pytest.param("hello", "", False, id="aligned"),
    pytest.param("hello", "", True, id="unaligned"),
    pytest.param("", "hello", False, id="late-aligned-text"),
    pytest.param("", "hello", True, id="late-unaligned-text"),
    pytest.param("", "", True, id="audio-only"),
]


def _assert_released(session, response_id):
    item_id = f"item_{response_id}"
    assert response_id not in session._playback.by_response
    assert response_id not in session._conversation.assistant_response_snapshots
    for entries in (
        session._conversation.pending_item_ids,
        session._conversation.pending_truncations_ms,
        session._conversation.hard_truncations_ms,
        session._conversation.history_item_placeholders,
    ):
        assert item_id not in entries


@pytest.mark.parametrize("initial_text,terminal_text,unaligned", _CONTENT)
# A zero cursor sends no initial ACK, so its before/after ordering is identical.
@pytest.mark.parametrize("ack_before_end,ack_ms", [(False, 0), (False, 200), (False, 400), (True, 200), (True, 400)])
@pytest.mark.parametrize("terminal_audio_ms", [400, 800])
def test_drain_finalization_event_orders(
    initial_text, terminal_text, unaligned, ack_before_end, ack_ms, terminal_audio_ms
):
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    first_user = session.commit_audio_input(transcript="first question")
    first = session.begin_response(turn_id=1)
    session.append_assistant_text(initial_text)
    session.mark_audio_sent(400, text_requires_complete_audio=unaligned)
    session.bind_draining_request("tts-first", first)
    session.snapshot_active_response_for_drain()
    second_user = session.commit_audio_input(transcript="next question")
    second = session.begin_response(turn_id=2)
    session.mark_audio_sent(100)
    second_playback = session.playback

    if ack_before_end and ack_ms:
        assert (
            apply_playback_ack(session, {"response_id": first, "played_ms": ack_ms})[0].type == "playback.acknowledged"
        )
        assert first in session._conversation.assistant_response_snapshots
    session.append_draining_assistant_text(first, terminal_text)
    session.mark_audio_sent(terminal_audio_ms, response_id=first, audio_complete=True)
    session.pop_draining_request("tts-first")
    session.release_finished_drain_response(first)
    if not ack_before_end and ack_ms:
        assert (
            apply_playback_ack(session, {"response_id": first, "played_ms": ack_ms})[0].type == "playback.acknowledged"
        )

    if ack_ms < terminal_audio_ms:
        assert first in session._conversation.assistant_response_snapshots
        assert session.playback_for_response(first).committed_ms == ack_ms
        if unaligned:
            assert all(message.get("content") != "hello" for message in session.history)
        assert (
            apply_playback_ack(session, {"response_id": first, "played_ms": terminal_audio_ms})[0].type
            == "playback.acknowledged"
        )
    # The caught-up-before-end case deliberately sends no second ACK.
    assert session.history == (
        first_user.message,
        {"role": "assistant", "content": initial_text + terminal_text},
        second_user.message,
    )
    _assert_released(session, first)
    assert session.active_response_id == second
    assert session.playback == second_playback


@pytest.mark.parametrize("delete_before_end", [False, True])
def test_drain_terminal_with_no_audio_or_deleted_item_does_not_create_history(delete_before_end):
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    first = session.begin_response()
    session.append_assistant_text("never heard")
    if delete_before_end:
        session.mark_audio_sent(400)
    session.bind_draining_request("tts-first", first)
    session.snapshot_active_response_for_drain()
    second = session.begin_response()
    if delete_before_end:
        assert session.delete_history_item(f"item_{first}")
        session.append_draining_assistant_text(first, " must stay deleted")
    session.pop_draining_request("tts-first")
    session.release_finished_drain_response(first)
    assert session.history == ()
    _assert_released(session, first)
    assert session.active_response_id == second


@pytest.mark.parametrize("terminal_kind", ["done", "cancel", "clear", "barge_in"])
@pytest.mark.parametrize("initial_text,terminal_text,unaligned", _CONTENT)
@pytest.mark.asyncio
async def test_runner_finalizes_caught_up_drain_without_another_ack(
    terminal_kind, initial_text, terminal_text, unaligned
):
    from dataclasses import replace

    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex import commands

    h = await open_harness()
    try:
        session = h.session
        session.capabilities = replace(session.capabilities, supports_concurrent_turn_requests=True)
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        first_user = session.commit_audio_input(transcript="first question")
        first = session.begin_response(turn_id=1)
        h.runner.emit(h.runner.model.response_created_payload(first, epoch=session.epoch))
        session.append_assistant_text(initial_text)
        session.mark_audio_sent(400, text_requires_complete_audio=unaligned)
        session.bind_draining_request("tts-first", first)
        session.snapshot_active_response_for_drain()
        second_user = session.commit_audio_input(transcript="next question")
        second = session.begin_response(turn_id=2)
        h.runner.emit(h.runner.model.response_created_payload(second, epoch=session.epoch))
        session.mark_audio_sent(100)
        events = await h.run(commands.AckPlayback(played_ms=400, response_id=first))
        assert not any(event.type == "error" for event in events)
        assert first in session._conversation.assistant_response_snapshots

        if terminal_kind == "done":
            await h.runner.model._send_one_model_output_event(
                {
                    "data_plane_request_id": "tts-first",
                    "end_of_turn": True,
                    "model_turn_id": 1,
                    "text": terminal_text,
                    "stage_role": "tts",
                    "audio_text_mark": False,
                    "text_requires_complete_audio": unaligned,
                    # No audio, audio duration, or audio_complete flag on this terminal.
                }
            )
            events = await h.settle()
            done = [event for event in events if event.type == "response.done"]
            assert len(done) == 1 and done[0].response_id == first
            assert session.history == (
                first_user.message,
                {"role": "assistant", "content": initial_text + terminal_text},
                second_user.message,
            )
            assert session.active_response_id == second
            assert session.playback.sent_ms == 100
        else:
            command = {
                "cancel": commands.CancelResponse,
                "clear": commands.ClearOutputAudio,
                "barge_in": commands.BargeIn,
            }[terminal_kind]()
            events = await h.run(command)
            assert not any(event.type == "error" for event in events)
            # Cancelling unfinished, unaligned audio cannot credit its entire text.
            heard = initial_text if not unaligned else ""
            assert all(
                message.get("content") in (heard, "")
                for message in session.history
                if message.get("role") == "assistant"
            )
        _assert_released(session, first)
    finally:
        await close_harness(h)


def test_drain_finalization_respects_hard_truncate():
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    first = session.begin_response()
    session.append_assistant_text("hello")
    session.mark_audio_sent(400)
    session.bind_draining_request("tts-first", first)
    session.snapshot_active_response_for_drain()
    session.begin_response()
    apply_playback_ack(session, {"response_id": first, "played_ms": 400, "committed_ms": 200, "truncate": True})
    session.pop_draining_request("tts-first")
    session.release_finished_drain_response(first)
    assert session.history == ({"role": "assistant", "content": "he"},)
    _assert_released(session, first)
    # Playback has stopped: no second ACK is required for retirement.
    apply_playback_ack(session, {"response_id": first, "played_ms": 400})
    assert session.history == ({"role": "assistant", "content": "he"},)
    _assert_released(session, first)


@pytest.mark.parametrize("initial_text,terminal_text,unaligned", _CONTENT)
@pytest.mark.parametrize("ack_before_end", [False, True])
@pytest.mark.asyncio
async def test_active_response_completion_joins_ack_and_terminal(
    initial_text, terminal_text, unaligned, ack_before_end
):
    from dataclasses import replace

    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex import commands

    h = await open_harness()
    try:
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        user = session.commit_audio_input(transcript="question")
        response_id = session.begin_response(turn_id=1)
        session.append_assistant_text(initial_text)
        session.mark_audio_sent(400, text_requires_complete_audio=unaligned)
        if ack_before_end:
            events = await h.run(commands.AckPlayback(played_ms=400, response_id=response_id))
            assert not any(event.type == "error" for event in events)
        await h.runner.model._send_one_model_output_event(
            {
                "end_of_turn": True,
                "model_turn_id": 1,
                "text": terminal_text,
                "stage_role": "tts",
                "audio_text_mark": False,
                "audio_complete": True,
                "text_requires_complete_audio": unaligned,
            }
        )
        await h.settle()
        if not ack_before_end:
            events = await h.run(commands.AckPlayback(played_ms=400, response_id=response_id))
            assert not any(event.type == "error" for event in events)
        assert session.history == (user.message, {"role": "assistant", "content": initial_text + terminal_text})
        _assert_released(session, response_id)
    finally:
        await close_harness(h)


@pytest.mark.parametrize("command_name", ["cancel", "clear", "barge_in"])
@pytest.mark.parametrize("unaligned", [False, True])
@pytest.mark.asyncio
async def test_active_cancel_cleans_fully_acked_state_without_crediting_unfinished_text(command_name, unaligned):
    from dataclasses import replace

    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex import commands

    h = await open_harness()
    try:
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        first = session.begin_response()
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_requires_complete_audio=unaligned)
        await h.run(commands.AckPlayback(played_ms=400, response_id=first))
        command = {"cancel": commands.CancelResponse, "clear": commands.ClearOutputAudio, "barge_in": commands.BargeIn}[
            command_name
        ]()
        events = await h.run(command)
        assert not any(event.type == "error" for event in events)
        assert session.history == ({"role": "assistant", "content": "" if unaligned else "hello"},)
        _assert_released(session, first)
    finally:
        await close_harness(h)


def test_multiple_draining_requests_do_not_finalize_each_other():
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    first = session.begin_response()
    session.append_assistant_text("hello")
    session.mark_audio_sent(400, text_requires_complete_audio=True)
    session.bind_draining_request("tts-first", first)
    session.bind_draining_request("tts-second", first)
    session.snapshot_active_response_for_drain()
    session.begin_response()
    apply_playback_ack(session, {"response_id": first, "played_ms": 400})
    session.pop_draining_request("tts-first")
    session.release_finished_drain_response(first)
    assert first in session._conversation.assistant_response_snapshots
    assert not session.playback_for_response(first).audio_complete
    session.pop_draining_request("tts-second")
    session.release_finished_drain_response(first)
    assert session.history == ({"role": "assistant", "content": "hello"},)
    _assert_released(session, first)
    session.release_finished_drain_response(first)
    assert session.history == ({"role": "assistant", "content": "hello"},)
    _assert_released(session, first)
