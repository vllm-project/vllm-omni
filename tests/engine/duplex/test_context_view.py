# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

import pytest

from vllm_omni.engine.duplex.config import DuplexContextViewPolicy, DuplexPlaybackCommitPolicy, DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLedger, DuplexContextLevel, source_refs
from vllm_omni.engine.duplex.session.context_view import DuplexHistoryView, render_history_view
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def session() -> DuplexEngineSession:
    return DuplexEngineSession(session_id="view-test", config=DuplexSessionConfig(model="test-model"))


def _audio(transcript: str | None = None) -> dict[str, object]:
    part: dict[str, object] = {"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}
    if transcript is not None:
        part["transcript"] = transcript
    return part


def _image() -> dict[str, object]:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def _text(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


def _user(*parts: dict[str, object]) -> dict[str, object]:
    return {"role": "user", "content": list(parts)}


def _assistant(text: str) -> dict[str, object]:
    return {"role": "assistant", "content": [_text(text)]}


def _render(session: DuplexEngineSession, ledger: DuplexContextLedger, keep: int) -> DuplexHistoryView:
    return render_history_view(session.history_snapshot(), ledger, DuplexContextViewPolicy(keep_media_user_turns=keep))


def test_l1_drops_audio_keeps_transcript(session):
    session.register_history_item("u1", _user(_audio("one")))
    session.register_history_item("a1", _assistant("reply"))

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages[0] == {"role": "user", "content": [_text("one")]}
    assert view.node_ids[0] is not None
    assert view.messages[1] == _assistant("reply")
    assert view.node_ids[1] is None


def test_bulk_l1_creation_scans_sources_a_constant_number_of_times(session):
    class CountedSources(tuple):
        visited: int = 0

        def __iter__(self):
            for source_id in super().__iter__():
                self.visited += 1
                yield source_id

    for index in range(50):
        session.register_history_item(f"u{index}", _user(_text("question"), _image()))
        session.register_history_item(f"a{index}", _assistant("answer"))
    original = session.history_snapshot()
    sources = CountedSources(original.source_ids)
    sources.visited = 0
    snapshot = replace(original, source_ids=sources)
    ledger = DuplexContextLedger()

    view = render_history_view(snapshot, ledger, DuplexContextViewPolicy(keep_media_user_turns=1))

    assert len(ledger.nodes()) == 49
    assert view.messages[0]["content"] == [_text("question")]
    assert view.messages[-2] == original.messages[-2]
    assert sources.visited <= 4 * len(snapshot.messages)


def test_l1_drops_image_keeps_text_parts_in_order(session):
    session.register_history_item("u1", _user(_text("a"), _image(), _text("b")))

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages[0]["content"] == [_text("a"), _text("b")]


def test_l1_does_not_duplicate_transcript_already_in_text(session):
    message = _user(_text("one"), _audio("one"))
    message["transcript"] = "one"
    session.register_history_item("u1", message)

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages[0] == {"role": "user", "content": [_text("one")], "transcript": "one"}


def test_l1_falls_back_to_message_transcript(session):
    message = _user(_audio())
    message["transcript"] = "from message"
    session.register_history_item("u1", message)

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages[0]["content"] == [_text("from message")]


def test_l1_leaves_media_without_text_verbatim(session):
    session.register_history_item("u1", _user(_audio()))
    snapshot = session.history_snapshot()

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages == snapshot.messages
    assert view.node_ids == (None,)


def test_l1_drops_old_image_item_when_run_keeps_audio(session):
    session.register_history_item("u1_image", _user(_image()))
    session.register_history_item("u1", _user(_audio("hello")))
    session.register_history_item("a1", _assistant("hi"))

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages[0] == {"role": "user", "content": []}
    assert view.messages[1] == {"role": "user", "content": [_text("hello")]}
    assert view.node_ids[0] is not None and view.node_ids[1] is not None
    assert len(view.messages) == 3


@pytest.mark.parametrize(
    "run",
    [
        [_user(_image())],
        [_user(_image()), _user(_audio())],
    ],
    ids=["image_only", "image_and_untranscribed_audio"],
)
def test_l1_keeps_image_item_when_run_would_be_empty(session, run):
    for index, message in enumerate(run):
        session.register_history_item(f"u{index}", message)
    session.register_history_item("a1", _assistant("hi"))
    snapshot = session.history_snapshot()

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages == snapshot.messages
    assert set(view.node_ids) == {None}


def test_l1_never_rewrites_assistant_system_tool(session):
    session.append_history_message({"role": "system", "content": [_text("rules"), _image()]})
    session.append_history_message({"role": "assistant", "content": [_text("look"), _audio("spoken")]})
    session.append_history_message({"role": "tool", "content": [_image()], "tool_call_id": "call-1"})
    snapshot = session.history_snapshot()

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages == snapshot.messages
    assert view.node_ids == (None, None, None)


def test_string_or_missing_content_is_verbatim(session):
    session.append_history_message({"role": "user", "content": "typed text"})
    session.append_history_message({"role": "user"})
    snapshot = session.history_snapshot()

    view = _render(session, DuplexContextLedger(), 0)

    assert view.messages == snapshot.messages
    assert view.node_ids == (None, None)


@pytest.mark.parametrize(
    ("keep", "derived"),
    [(0, {0, 1, 3, 5}), (1, {0, 1, 3}), (2, {0, 1}), (3, set())],
)
def test_recent_user_turns_stay_verbatim(session, keep, derived):
    session.register_history_item("u1_image", _user(_image()))
    session.register_history_item("u1", _user(_audio("one")))
    session.register_history_item("a1", _assistant("first"))
    session.register_history_item("u2", _user(_audio("two")))
    session.register_history_item("a2", _assistant("second"))
    session.register_history_item("u3", _user(_audio("three")))
    snapshot = session.history_snapshot()

    view = _render(session, DuplexContextLedger(), keep)

    assert {index for index, node_id in enumerate(view.node_ids) if node_id is not None} == derived
    for index in set(range(len(snapshot.messages))) - derived:
        assert view.messages[index] == snapshot.messages[index]


def test_view_aligned_one_to_one_with_snapshot(session):
    session.register_history_item("u1_image", _user(_image()))
    session.register_history_item("u1", _user(_audio("one")))
    session.register_history_item("a1", _assistant("first"))
    session.append_history_message({"role": "user", "content": "typed"})
    session.register_history_item("u2", _user(_audio()))
    snapshot = session.history_snapshot()

    view = _render(session, DuplexContextLedger(), 1)

    assert len(view.messages) == len(view.node_ids) == len(snapshot.messages)
    assert [message["role"] for message in view.messages] == [message["role"] for message in snapshot.messages]
    assert view.source_ids == snapshot.source_ids


def test_transcript_update_rederives_l1_node(session):
    session.register_history_item("u0_image", _user(_text("look"), _image()))
    committed = session.commit_audio_input()
    session.register_history_item("u1", committed.message)
    part = session.input_audio_part("u1")
    ledger = DuplexContextLedger()

    before = _render(session, ledger, 0)
    assert before.messages[1] == {"role": "user", "content": []}

    assert session.update_input_transcript("u1", "heard words", audio_part=part)
    after = _render(session, ledger, 0)

    assert after.node_ids[0] == before.node_ids[0]
    assert after.node_ids[1] is not None and before.node_ids[1] is not None
    assert after.node_ids[1] > before.node_ids[1]
    assert after.messages[1]["content"] == [_text("heard words")]


def test_rerender_reuses_valid_nodes(session):
    session.register_history_item("u1", _user(_audio("one")))
    ledger = DuplexContextLedger()

    first = _render(session, ledger, 0)
    second = _render(session, ledger, 0)

    assert first.node_ids == second.node_ids
    assert len(ledger.nodes()) == 1


def test_view_messages_are_detached_from_ledger(session):
    session.register_history_item("u1", _user(_audio("one")))
    session.register_history_item("u2", _user(_audio("two")))
    session.register_history_item("a1", _assistant("reply"))
    ledger = DuplexContextLedger()
    first = _render(session, ledger, 0)
    expected = _render(session, ledger, 0).messages

    first.messages[0]["content"].append(_text("injected"))
    first.messages[2]["content"][0]["text"] = "edited"

    assert _render(session, ledger, 0).messages == expected
    assert session.history_snapshot().messages[2] == _assistant("reply")


@pytest.mark.parametrize("value", [-1, True, "2", 1.5])
def test_policy_rejects_invalid_keep_media_user_turns(value):
    with pytest.raises(ValueError, match="keep_media_user_turns"):
        DuplexContextViewPolicy(keep_media_user_turns=value)


KEEP_NONE = DuplexContextViewPolicy(keep_media_user_turns=0)


def _ack_only(session: DuplexEngineSession) -> None:
    session.replace_config(
        DuplexSessionConfig(model="test-model", playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value)
    )


def _half_played_reply(session: DuplexEngineSession, text: str = "hello world") -> str:
    response_id = session.begin_response()
    session.append_assistant_text(text)
    session.mark_audio_sent(duration_ms=2000, text_chars=len(text))
    session.acknowledge_playback(played_ms=1000)
    session.register_history_item(f"item_{response_id}", session.end_response())
    return response_id


def _summary_over(session: DuplexEngineSession, source_index: int, *parents: int) -> int:
    snapshot = session.history_snapshot()
    return session.context_ledger.add_node(
        snapshot,
        level=DuplexContextLevel.SUMMARY,
        sources=source_refs(snapshot, snapshot.source_ids[source_index]),
        parents=parents,
        message={"role": "system", "content": "summary"},
    ).node_id


def test_refresh_and_render_do_not_change_history_revision(session):
    session.register_history_item("u1", _user(_audio("one")))
    session.register_history_item("u2", _user(_audio("two")))
    before = session.history_snapshot()

    view = session.render_history_view(KEEP_NONE)
    session.render_history_view(KEEP_NONE)

    assert view.node_ids[0] is not None
    assert session.history_snapshot() == before


def test_epoch_change_alone_keeps_nodes(session):
    session.register_history_item("u1", _user(_audio("one")))
    before = session.render_history_view(KEEP_NONE)

    session.barge_in()

    assert session.render_history_view(KEEP_NONE).node_ids == before.node_ids


def test_ledger_survives_resume_lease(session):
    session.register_history_item("u1", _user(_audio("one")))
    before = session.render_history_view(KEEP_NONE)
    session.detach_lease()

    assert session.resume_lease(expected_lease_generation=0) == 1
    assert session.render_history_view(KEEP_NONE).node_ids == before.node_ids


def test_view_reflects_heard_prefix(session):
    _ack_only(session)
    session.register_history_item("u1", _user(_audio("question")))
    response_id = _half_played_reply(session)

    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    view = session.render_history_view(KEEP_NONE)

    assert view.messages[1] == {"role": "assistant", "content": "hell"}
    assert view.node_ids[1] is None


def test_heard_prefix_invalidates_dependent_summary(session):
    _ack_only(session)
    session.register_history_item("u1", _user(_audio("question")))
    response_id = _half_played_reply(session)
    l1_parent = session.render_history_view(KEEP_NONE).node_ids[0]
    assert l1_parent is not None
    summary = _summary_over(session, 1, l1_parent)

    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    session.render_history_view(KEEP_NONE)

    assert not session.context_ledger.valid(summary)
    assert session.context_ledger.valid(l1_parent)


def test_hard_cut_invalidates_dependents(session):
    session.register_history_item("u1", _user(_audio("question")))
    response_id = session.begin_response()
    session.append_assistant_text("hello worlds")
    session.mark_audio_sent(duration_ms=2000, text_chars=12)
    session.acknowledge_playback(played_ms=2000)
    item_id = f"item_{response_id}"
    session.register_history_item(item_id, session.end_response())
    l1_parent = session.render_history_view(KEEP_NONE).node_ids[0]
    assert l1_parent is not None
    summary = _summary_over(session, 1, l1_parent)

    assert session.truncate_history_item(item_id, audio_end_ms=1000, hard=True)
    session.render_history_view(KEEP_NONE)

    assert not session.context_ledger.valid(summary)
    assert session.context_ledger.valid(l1_parent)


def test_heard_prefix_does_not_touch_other_sources_nodes(session):
    _ack_only(session)
    session.register_history_item("u1", _user(_audio("one")))
    session.register_history_item("u2", _user(_audio("two")))
    response_id = _half_played_reply(session)
    before = session.render_history_view(KEEP_NONE)

    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    after = session.render_history_view(KEEP_NONE)

    assert after.node_ids == before.node_ids
    assert after.messages[:2] == before.messages[:2]


def test_deletion_moves_message_back_into_verbatim_tail(session):
    session.register_history_item("u1", _user(_audio("one")))
    session.register_history_item("a1", _assistant("first"))
    session.register_history_item("u2", _user(_audio("two")))
    keep_one = DuplexContextViewPolicy(keep_media_user_turns=1)
    before = session.render_history_view(keep_one)
    assert before.node_ids[0] is not None and before.node_ids[2] is None

    assert session.delete_history_item("u2")
    after = session.render_history_view(keep_one)

    assert after.node_ids == (None, None)
    assert after.messages[0] == _user(_audio("one"))
