# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stopped playback retires evidence; stored answers settle their tool owner."""

from dataclasses import replace

import pytest

from tests.engine.duplex.test_engine_session import _session
from tests.engine.duplex.test_playback_finalization import _CONTENT, _assert_released
from tests.engine.duplex.test_session_runner import close_harness
from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("initial_text,terminal_text,unaligned", _CONTENT)
@pytest.mark.parametrize("draining", [False, True])
@pytest.mark.parametrize("truncate_before_end", [False, True])
@pytest.mark.parametrize("cutoff", [0, 200, 400])
def test_hard_truncate_retires_output_without_another_ack(
    initial_text, terminal_text, unaligned, draining, truncate_before_end, cutoff
):
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    response = session.begin_response()
    session.append_assistant_text(initial_text)
    session.mark_audio_sent(400, text_requires_complete_audio=unaligned)
    if draining:
        session.bind_draining_request("tts", response)
        session.snapshot_active_response_for_drain()
        next_response = session.begin_response()
        session.mark_audio_sent(100)
    ack = {"response_id": response, "played_ms": cutoff, "truncate": True}
    if truncate_before_end:
        assert apply_playback_ack(session, ack)[0].type == "playback.acknowledged"
        assert response in session._playback.by_response
    if draining:
        session.append_draining_assistant_text(response, terminal_text)
        session.pop_draining_request("tts")
        session.release_finished_drain_response(response)
    else:
        session.append_assistant_text(terminal_text)
        session.mark_audio_sent(400, audio_complete=True)
        message = session.end_response()
        session.register_history_item(f"item_{response}", message)
        session.finalize_response_playback(response)
    if not truncate_before_end:
        assert apply_playback_ack(session, ack)[0].type == "playback.acknowledged"
    expected = "" if unaligned and cutoff < 400 else (initial_text + terminal_text)[: 5 * cutoff // 400]
    assert session.history == ({"role": "assistant", "content": expected},)
    _assert_released(session, response)
    if draining:
        assert session.active_response_id == next_response
        assert session.playback.sent_ms == 100
    else:
        assert session.playback.sent_ms == session.playback.committed_ms
    # The stop is permanent even if the browser sends a duplicate/stale full ACK.
    apply_playback_ack(session, {"response_id": response, "played_ms": 400})
    assert session.history == ({"role": "assistant", "content": expected},)
    _assert_released(session, response)
    # A stricter stop after full evidence was released cannot restore old text.
    apply_playback_ack(session, {"response_id": response, "played_ms": 0, "truncate": True})
    assert session.history == ({"role": "assistant", "content": ""},)
    _assert_released(session, response)


@pytest.mark.parametrize("command_type", [commands.CancelResponse, commands.ClearOutputAudio, commands.BargeIn])
@pytest.mark.parametrize("ack_ms", [0, 200, 400])
@pytest.mark.parametrize("new_input", [False, True])
@pytest.mark.parametrize("unaligned", [False, True])
@pytest.mark.asyncio
async def test_cancel_settles_only_the_answer_actually_stored(command_type, ack_ms, new_input, unaligned):
    h = await open_qwen()
    try:
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        pair = {"response_id": response, "call_id": "call-a", "messages": []}
        unrelated = {"response_id": "other-response", "call_id": "call-b", "messages": []}
        session.replace_runtime_config(
            {
                **session.runtime_config,
                "qwen_tool_turns_inflight": [pair],
                "qwen_tool_turns": [unrelated],
                "qwen_tool_response_id": response,
            }
        )
        session.mark_audio_sent(400, text_requires_complete_audio=unaligned)
        await h.run(commands.AckPlayback(played_ms=ack_ms, response_id=response))
        session.append_assistant_text("hello")
        user = session.commit_audio_input(transcript="next question") if new_input else None
        events = await h.run(command_type())
        assert not any(event.type == "error" for event in events)
        expected = "" if unaligned else "hello"[: 5 * ack_ms // 400]
        assistant = ({"role": "assistant", "content": expected},) if unaligned or ack_ms else ()
        assert session.history == assistant + ((user.message,) if user else ())
        assert session.runtime_config.get("qwen_tool_turns") == ([unrelated] if expected else [pair, unrelated])
        assert not session.runtime_config.get("qwen_tool_turns_inflight")
        if ack_ms == 400:
            _assert_released(session, response)
        else:
            assert response in session._conversation.assistant_response_snapshots
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_truncate_item", [False, True])
@pytest.mark.parametrize("calibration_enabled", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_truncation_preserves_bounded_calibration_owner(use_truncate_item, calibration_enabled, cancelled):
    from tests.engine.duplex.test_session_runner import append_audio, tts_output

    async def calibrate(snapshot):
        assert snapshot.played_ms == 200
        return 2

    h = await open_qwen(calibrate=calibrate if calibration_enabled else None)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(final=True, create_response=True))
        session = h.session
        request = h.port.submissions[-1].context.request_id
        response = session.active_response_id
        await h.deliver_and_settle(tts_output(request, samples=0, text="hello", finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request, samples=9600, finished=not cancelled), stage_id=2)
        if cancelled:
            await h.run(commands.CancelResponse())
        if use_truncate_item:
            events = await h.run(commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=200))
        else:
            events = await h.run(
                commands.command_from_realtime(
                    {
                        "type": "playback.ack",
                        "response_id": response,
                        "played_ms": 200,
                        "truncate": True,
                    }
                )
            )
        assert not any(event.type == "error" for event in events)
        if calibration_enabled:
            controller = h.runner.ctx.history_calibration
            await controller.before_prompt()
            assert session.history[-1]["content"] == "he"
            assert response in session._conversation.assistant_response_snapshots
            # Evidence belongs to the bounded calibration cache, not a future ACK.
            controller.discard(response)
        else:
            assert session.history[-1]["content"] == ""
        _assert_released(session, response)
        before = session.history
        events = await h.run(commands.AckPlayback(response_id=response, played_ms=400))
        ack = next(event for event in events if event.type == "playback.acknowledged")
        assert ack.details["history_committed"] is False
        assert session.history == before
        _assert_released(session, response)
    finally:
        await close_harness(h)


@pytest.mark.parametrize("cutoff", [0, 200])
def test_hard_stop_tightens_text_already_stored_during_drain(cutoff):
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    response = session.begin_response()
    session.append_assistant_text("hello")
    session.mark_audio_sent(400)
    session.bind_draining_request("tts", response)
    session.snapshot_active_response_for_drain()
    session.begin_response()
    apply_playback_ack(session, {"response_id": response, "played_ms": 400})
    assert session.history == ({"role": "assistant", "content": "hello"},)
    apply_playback_ack(session, {"response_id": response, "played_ms": cutoff, "truncate": True})
    expected = "hello"[: 5 * cutoff // 400]
    assert session.history == ({"role": "assistant", "content": expected},)
    session.pop_draining_request("tts")
    session.release_finished_drain_response(response)
    assert session.history == ({"role": "assistant", "content": expected},)
    _assert_released(session, response)


def test_hard_stop_retires_registered_item_without_snapshot():
    session = _session(config=DuplexSessionConfig(playback_commit_policy="ack_only"))
    session.register_history_item("item_old", {"role": "assistant", "content": "hello"})
    session.mark_audio_sent(400, response_id="old")
    apply_playback_ack(session, {"response_id": "old", "played_ms": 200, "truncate": True})
    assert session.history == ({"role": "assistant", "content": "he"},)
    _assert_released(session, "old")
    apply_playback_ack(session, {"response_id": "old", "played_ms": 400})
    assert session.history == ({"role": "assistant", "content": "he"},)
    _assert_released(session, "old")


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_hard,initial_ms", [(False, 400), (True, 400), (True, 300), (True, 200)])
@pytest.mark.parametrize("next_response", [False, True])
@pytest.mark.parametrize("partial_marks", [False, True])
async def test_repeated_item_truncation_keeps_original_alignment(
    initial_hard, initial_ms, next_response, partial_marks
):
    h = await open_qwen()
    try:
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        if partial_marks:
            session.mark_audio_sent(200, text_chars=2)
            session.mark_audio_sent(400, audio_complete=True)
        else:
            session.mark_audio_sent(400, text_chars=5, audio_complete=True)
        session.register_history_item(f"item_{response}", session.end_response())
        await h.runner._run_internal_payload(
            {"type": "playback.ack", "response_id": response, "played_ms": initial_ms, "truncate": initial_hard}
        )
        await h.settle()
        assert session.history[-1]["content"] == "hello"[: 5 * initial_ms // 400]
        _assert_released(session, response)
        assert f"item_{response}" not in session._conversation.heard_prefixes
        assert session._conversation.item_audio_text_marks[f"item_{response}"]
        if next_response:
            newer = session.begin_response()
            h.runner.emit(h.runner.model.response_created_payload(newer, epoch=session.epoch))
            # This response has no alignment; its cursor must not affect the old item.
            session.mark_audio_sent(1000, text_requires_complete_audio=True)
        playback = session.playback
        expected = session.history[-1]["content"]
        for cutoff, keep_chars in [(300, 3), (200, 2), (160, 1 if partial_marks else 2), (0, 0)]:
            expected = expected[:keep_chars]
            events = await h.run(commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=cutoff))
            assert not any(event.type == "error" for event in events)
            assert session.history[-1]["content"] == expected
            for ack_ms in (400, 100, 0, 400):
                events = await h.run(commands.AckPlayback(response_id=response, played_ms=ack_ms))
                assert not any(event.type == "error" for event in events)
                ack = next(event for event in events if event.type == "playback.acknowledged")
                assert ack.details["history_committed"] is False
                assert session.history[-1]["content"] == expected
                _assert_released(session, response)
                assert session.playback == playback
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("unaligned", [False, True])
@pytest.mark.parametrize("new_output", [None, "aligned", "unaligned"])
@pytest.mark.parametrize("cutoff", [1, 200])
async def test_finalized_markless_item_cannot_borrow_another_responses_alignment(unaligned, new_output, cutoff):
    h = await open_qwen()
    try:
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        user = session.commit_audio_input(transcript="first question")
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        session.mark_audio_sent(400, text_requires_complete_audio=unaligned, audio_complete=True)
        session.register_history_item(f"item_{response}", session.end_response())
        await h.run(commands.AckPlayback(response_id=response, played_ms=400))
        assert session.history == (user.message, {"role": "assistant", "content": "hello"})
        _assert_released(session, response)
        next_user = session.commit_audio_input(transcript="next question")
        if new_output is not None:
            newer = session.begin_response()
            h.runner.emit(h.runner.model.response_created_payload(newer, epoch=session.epoch))
            session.append_assistant_text("a different answer")
            session.mark_audio_sent(
                1000,
                text_chars=18 if new_output == "aligned" else None,
                text_requires_complete_audio=new_output == "unaligned",
            )
        active_playback = session.playback
        events = await h.run(commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=cutoff))
        assert not any(event.type == "error" for event in events), events
        # With neither timing marks nor duration left, no prefix is provable.
        # Truncation preserves the turn and the follow-up playback ACK target.
        expected = (user.message, {"role": "assistant", "content": ""}, next_user.message)
        assert session.history == expected
        assert f"item_{response}" in session.history_item_ids
        ack = await h.run(commands.AckPlayback(response_id=response, played_ms=400))
        assert not any(event.type == "error" for event in ack)
        assert session.history == expected
        assert session.playback == active_playback
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["aligned", "markless", "unaligned"])
@pytest.mark.parametrize("hard_stop", [False, True])
async def test_finalized_ack_does_not_report_a_commit_or_resettle_tools(monkeypatch, output, hard_stop):
    from vllm_omni.engine.duplex.session import runner as runner_module

    calls = []
    settle = runner_module.settle_response_runtime

    def record_settlement(*args, **kwargs):
        calls.append(kwargs)
        return settle(*args, **kwargs)

    monkeypatch.setattr(runner_module, "settle_response_runtime", record_settlement)
    h = await open_qwen()
    try:
        session = h.session
        session.replace_config(replace(session.config, playback_commit_policy="ack_only"))
        response = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("hello")
        session.mark_audio_sent(
            400,
            text_chars=5 if output == "aligned" else None,
            text_requires_complete_audio=output == "unaligned",
            audio_complete=True,
        )
        session.register_history_item(f"item_{response}", session.end_response())
        session.replace_runtime_config(
            {
                **session.runtime_config,
                "qwen_tool_turns_inflight": [{"response_id": response, "call_id": "old-tool", "messages": []}],
                "qwen_tool_response_id": response,
            }
        )
        await h.runner._run_internal_payload(
            {
                "type": "playback.ack",
                "response_id": response,
                "played_ms": 400,
                "truncate": hard_stop,
            }
        )
        await h.settle()
        assert len(calls) == 1
        assert not session.runtime_config.get("qwen_tool_turns_inflight")
        _assert_released(session, response)
        newer = session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(newer, epoch=session.epoch))
        session.append_assistant_text("new text")
        session.mark_audio_sent(1000, text_chars=8)
        session.replace_runtime_config(
            {
                **session.runtime_config,
                "qwen_tool_turns_inflight": [{"response_id": newer, "call_id": "new-tool", "messages": []}],
                "qwen_tool_response_id": newer,
            }
        )
        runtime = dict(session.runtime_config)
        playback = session.playback
        for ack_ms in (400, 200, 0, 800):
            events = await h.run(commands.AckPlayback(response_id=response, played_ms=ack_ms))
            ack = next(event for event in events if event.type == "playback.acknowledged")
            assert ack.details["history_committed"] is False
            assert len(calls) == 1
            assert session.history == ({"role": "assistant", "content": "hello"},)
            assert session.runtime_config == runtime
            assert session.playback == playback
            _assert_released(session, response)
    finally:
        await close_harness(h)
