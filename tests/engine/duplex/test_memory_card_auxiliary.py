# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Only owner-bound, current and legally heard evidence can create cards."""

import gc
import weakref

import pytest

from vllm_omni.engine.duplex.config import DuplexPlaybackCommitPolicy, DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CITY = "user/self/current_city"


@pytest.fixture
def session() -> DuplexEngineSession:
    return DuplexEngineSession(session_id="memory-input", config=DuplexSessionConfig(model="test-model"))


def _user(session, item_id="u1", text="The user lives in Hangzhou."):
    session.register_history_item(item_id, {"role": "user", "content": text})


def _proposal(context, *, handle=0, start=0, end=None, text="Hangzhou"):
    from vllm_omni.engine.duplex.memory_types import (
        DuplexMemoryEvidenceSelection,
        DuplexMemoryProposal,
        DuplexMemoryProposalBatch,
    )

    if text is None:
        return DuplexMemoryProposalBatch(operations=())
    if end is None:
        end = len(context.evidence[0].text)
    return DuplexMemoryProposalBatch(
        operations=(
            DuplexMemoryProposal(
                fact_key=CITY,
                text=text,
                evidence=(DuplexMemoryEvidenceSelection(handle=handle, char_start=start, char_end=end),),
            ),
        )
    )


def _spoken(session):
    session.replace_config(
        DuplexSessionConfig(model="test-model", playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value)
    )
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11)
    session.acknowledge_playback(played_ms=1000)
    session.register_history_item(f"item_{response_id}", session.end_response())
    return response_id


def test_context_contains_detached_text_and_session_source_identity(session):
    _user(session)
    snapshot = session.history_snapshot()

    context = session.capture_memory_context()

    assert [(item.handle, item.role, item.text) for item in context.evidence] == [
        (0, "user", "The user lives in Hangzhou.")
    ]
    assert (context.evidence[0].source_id, context.evidence[0].revision) == (
        snapshot.source_ids[0],
        snapshot.source_revisions[0],
    )
    _user(session, text="The user lives in Shanghai.")
    assert context.evidence[0].text == "The user lives in Hangzhou."
    assert not session.memory_context_matches(context)


def test_context_excludes_pending_user_items_and_non_conversation_roles(session):
    _user(session, "old", "A committed old fact.")
    _user(session, "pending", "An unanswered question.")
    session.register_history_item("system", {"role": "system", "content": "An instruction, not a user fact."})
    session.register_history_item("tool", {"role": "tool", "content": "A tool payload."})

    context = session.capture_memory_context(excluded_item_ids=frozenset({"pending"}))

    assert [item.text for item in context.evidence] == ["A committed old fact."]


def test_context_keeps_available_transcript_without_retaining_media(session):
    session.register_history_item(
        "media",
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look here."},
                {"type": "audio_url", "audio_url": {"url": "fixture-audio"}, "transcript": "I live in Hangzhou."},
                {"type": "image_url", "image_url": {"url": "fixture-image"}},
            ],
        },
    )

    context = session.capture_memory_context()

    assert [item.text for item in context.evidence] == ["Look here.\nI live in Hangzhou."]
    assert "fixture-audio" not in repr(context)
    assert "fixture-image" not in repr(context)


def test_media_without_available_text_does_not_create_work(session):
    session.register_history_item(
        "media", {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "fixture"}}]}
    )

    assert session.capture_memory_context() is None


def test_context_does_not_keep_deleted_source_media_alive(session):
    class MediaPayload:
        pass

    payload = MediaPayload()
    retained = weakref.ref(payload)
    session.register_history_item(
        "media",
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A committed caption."},
                {"type": "image_url", "image_url": payload},
            ],
        },
    )
    context = session.capture_memory_context()
    session.delete_history_item("media")
    del payload
    gc.collect()

    assert retained() is None
    assert context.evidence[0].text == "A committed caption."


def test_capture_bounds_source_count_and_text_without_skipping_partial_source(session):
    _user(session, "a", "ABCD")
    _user(session, "b", "EFGH")
    _user(session, "c", "IJKL")

    first = session.capture_memory_context(max_sources=2, max_chars=6)

    assert [item.text for item in first.evidence] == ["ABCD", "EF"]
    session.admit_memory_result(context=first, proposal=_proposal(first, text=None))
    second = session.capture_memory_context(max_sources=2, max_chars=6)
    assert [item.text for item in second.evidence] == ["GH", "IJKL"]


def test_uncalibrated_incomplete_audio_is_not_extraction_evidence(session):
    _spoken(session)

    assert session.capture_memory_context() is None


def test_only_calibrated_heard_prefix_is_available_to_model(session):
    response_id = _spoken(session)
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)

    context = session.capture_memory_context()

    assert [item.text for item in context.evidence] == ["hell"]
    result = session.admit_memory_result(context=context, proposal=_proposal(context, end=11))
    assert result.status == "invalid"
    assert session.context_ledger.nodes() == ()


def test_retired_heard_prefix_remains_legal_evidence(session):
    response_id = _spoken(session)
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    session.retire_history_calibration(response_id)

    context = session.capture_memory_context()

    assert [item.text for item in context.evidence] == ["hell"]


def test_bound_text_only_assistant_is_eligible(session):
    session.register_history_item("text-only", {"role": "assistant", "content": "A text-only answer."})

    assert [item.text for item in session.capture_memory_context().evidence] == ["A text-only answer."]


def test_unbound_assistant_cannot_bypass_audio_provenance(session):
    session.append_history_message({"role": "assistant", "content": "Unbound output."})

    assert session.capture_memory_context() is None


def test_same_visible_prefix_with_new_cutoff_rejects_captured_context(session):
    response_id = _spoken(session)
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=900, char_end=4)
    context = session.capture_memory_context()
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)

    assert [item.text for item in context.evidence] == ["hell"]
    assert not session.memory_context_matches(context)


def test_unrelated_history_append_does_not_invalidate_recorded_sources(session):
    _user(session)
    context = session.capture_memory_context()
    _user(session, "later", "An unrelated later message.")

    assert session.memory_context_matches(context)


def test_internal_runtime_generation_does_not_fence_context(session):
    _user(session)
    context = session.capture_memory_context()
    session.replace_runtime_config({"private_tool_state": "changed"})

    assert session.memory_context_matches(context)


@pytest.mark.parametrize("change", ["epoch", "config", "source", "delete", "recreate", "close"])
def test_execution_or_source_change_rejects_result(session, change):
    _user(session)
    context = session.capture_memory_context()
    if change == "epoch":
        session.epoch += 1
    elif change == "config":
        session.replace_config(DuplexSessionConfig(model="changed-model"))
    elif change == "source":
        _user(session, text="Shanghai")
    elif change == "delete":
        session.delete_history_item("u1")
    elif change == "recreate":
        session.delete_history_item("u1")
        _user(session)
    else:
        session.state = DuplexSessionState.CLOSED

    result = session.admit_memory_result(context=context, proposal=_proposal(context))

    assert result.status == "stale"
    assert session.context_ledger.nodes() == ()


def test_equal_public_session_and_source_ids_do_not_authorize_foreign_context(session):
    _user(session)
    context = session.capture_memory_context()
    other = DuplexEngineSession(session_id=session.session_id, config=session.config)
    _user(other)

    assert not other.memory_context_matches(context)
    assert other.admit_memory_result(context=context, proposal=_proposal(context)).status == "stale"
    assert other.context_ledger.nodes() == ()


@pytest.mark.parametrize("handle", [-1, 999, False])
def test_model_handle_cannot_manufacture_a_source(session, handle):
    _user(session)
    context = session.capture_memory_context()

    result = session.admit_memory_result(context=context, proposal=_proposal(context, handle=handle))

    assert result.status == "invalid"
    assert session.context_ledger.nodes() == ()
    assert session.capture_memory_context().work_key == context.work_key


def test_successful_empty_result_advances_only_the_processed_span(session):
    _user(session, text="ABCDEFGH")
    first = session.capture_memory_context(max_chars=4)
    assert first.evidence[0].text == "ABCD"

    assert session.admit_memory_result(context=first, proposal=_proposal(first, text=None)).status == "no_change"
    second = session.capture_memory_context(max_chars=4)

    assert second.evidence[0].text == "EFGH"
    assert (second.evidence[0].char_start, second.evidence[0].char_end) == (4, 8)
    assert session.admit_memory_result(context=second, proposal=_proposal(second, text=None)).status == "no_change"
    assert session.capture_memory_context(max_chars=4) is None


def test_edit_behind_completed_cursor_requeues_current_evidence(session):
    _user(session)
    context = session.capture_memory_context()
    session.admit_memory_result(context=context, proposal=_proposal(context, text=None))
    assert session.capture_memory_context() is None
    _user(session, text="Now the user lives in Shanghai.")

    changed = session.capture_memory_context()

    assert [item.text for item in changed.evidence] == ["Now the user lives in Shanghai."]
    assert changed.work_key != context.work_key


def test_late_placeholder_fill_is_revisited_after_newer_source_was_processed(session):
    session.reserve_history_item("assistant-old")
    _user(session)
    context = session.capture_memory_context()
    session.admit_memory_result(context=context, proposal=_proposal(context, text=None))
    session.register_history_item("assistant-old", {"role": "assistant", "content": "Earlier text-only answer."})

    context = session.capture_memory_context()

    assert context.evidence[0].text == "Earlier text-only answer."


def test_model_candidates_bind_refs_without_changing_history(session):
    _user(session)
    context = session.capture_memory_context()
    before = session.history_snapshot()

    result = session.admit_memory_result(context=context, proposal=_proposal(context))

    assert result.status == "applied"
    assert result.node_ids == (1,)
    (node,) = session.context_ledger.card_candidates()
    assert [(ref.source_id, ref.revision) for ref in node.sources] == [
        (before.source_ids[0], before.source_revisions[0])
    ]
    assert session.history_snapshot() == before
    assert session.admit_memory_result(context=context, proposal=_proposal(context)).status == "stale"
