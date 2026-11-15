# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Authoritative source identity and detached duplex-history snapshots."""

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex.config import DuplexPlaybackCommitPolicy, DuplexSessionConfig
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def session() -> DuplexEngineSession:
    return DuplexEngineSession(session_id="history-test", config=DuplexSessionConfig(model="test-model"))


def test_snapshot_aligns_unbound_and_registered_sources(session):
    session.append_history_message({"role": "user", "content": "first"})
    session.register_history_item("public-second", {"role": "user", "content": "second"})

    snapshot = session.history_snapshot()

    assert snapshot.messages == (
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    )
    assert snapshot.item_ids == (None, "public-second")
    assert len(set(snapshot.source_ids)) == 2
    assert snapshot.source_revisions == (0, 0)
    assert snapshot.history_revision == session.history_revision


def test_old_snapshot_keeps_nested_content_after_source_replacement(session):
    session.register_history_item("user", {"role": "user", "content": [{"type": "text", "text": "before"}]})
    before = session.history_snapshot()

    session.register_history_item("user", {"role": "user", "content": [{"type": "text", "text": "after"}]})

    assert before.messages == ({"role": "user", "content": [{"type": "text", "text": "before"}]},)
    assert session.history_snapshot().messages == ({"role": "user", "content": [{"type": "text", "text": "after"}]},)


def test_mutating_snapshot_nested_media_does_not_change_authoritative_history(session):
    session.register_history_item(
        "image", {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image-before"}}]}
    )
    snapshot = session.history_snapshot()
    snapshot.messages[0]["content"][0]["image_url"]["url"] = "view-only"

    assert session.history_snapshot().messages == (
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image-before"}}]},
    )


def test_equal_messages_have_distinct_sources(session):
    session.register_history_item("one", {"role": "user", "content": "same"})
    session.register_history_item("two", {"role": "user", "content": "same"})

    snapshot = session.history_snapshot()

    assert snapshot.source_ids[0] != snapshot.source_ids[1]
    assert snapshot.item_ids == ("one", "two")


def test_append_preserves_existing_source_identity_and_revision(session):
    session.append_history_message({"role": "user", "content": "first"})
    before = session.history_snapshot()

    session.append_history_message({"role": "user", "content": "second"})
    after = session.history_snapshot()

    assert after.source_ids[0] == before.source_ids[0]
    assert after.source_revisions[0] == before.source_revisions[0]
    assert after.history_revision > before.history_revision


def test_hidden_placeholder_does_not_change_visible_history(session):
    before = session.history_snapshot()
    session.reserve_history_item("pending")
    pending = session.history_snapshot()

    assert pending.messages == pending.item_ids == pending.source_ids == pending.source_revisions == ()
    assert pending.history_revision == before.history_revision

    session.register_history_item("pending", {"role": "assistant", "content": "ready"})
    ready = session.history_snapshot()
    assert ready.item_ids == ("pending",)
    assert len(ready.source_ids) == 1
    assert ready.history_revision > pending.history_revision


def test_replacement_advances_only_changed_source(session):
    session.register_history_item("one", {"role": "user", "content": "old"})
    session.register_history_item("two", {"role": "user", "content": "unchanged"})
    before = session.history_snapshot()

    session.register_history_item("one", {"role": "user", "content": "new"})
    after = session.history_snapshot()

    assert after.source_ids == before.source_ids
    assert after.source_revisions == (before.source_revisions[0] + 1, before.source_revisions[1])
    assert after.history_revision > before.history_revision


def test_equal_value_replacement_is_not_a_source_change(session):
    session.register_history_item("one", {"role": "user", "content": [{"type": "text", "text": "same"}]})
    before = session.history_snapshot()

    session.register_history_item("one", {"role": "user", "content": [{"type": "text", "text": "same"}]})

    assert session.history_snapshot() == before


def test_binding_public_item_id_preserves_internal_source(session):
    committed = session.commit_audio_input(transcript="hello")
    before = session.history_snapshot()

    session.register_history_item("audio", committed.message)
    after = session.history_snapshot()

    assert after.messages == before.messages
    assert after.source_ids == before.source_ids
    assert after.item_ids == ("audio",)
    assert after.source_revisions[0] > before.source_revisions[0]


def test_asr_transcript_update_advances_source_and_preserves_old_snapshot(session):
    committed = session.commit_audio_input()
    session.register_history_item("audio", committed.message)
    part = session.input_audio_part("audio")
    before = session.history_snapshot()

    assert session.update_input_transcript("audio", "heard words", audio_part=part)
    after = session.history_snapshot()

    assert before.messages == (
        {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}]},
    )
    assert after.messages[0]["transcript"] == "heard words"
    assert after.messages[0]["content"][0]["transcript"] == "heard words"
    assert after.source_ids == before.source_ids
    assert after.source_revisions[0] == before.source_revisions[0] + 1


def test_stale_asr_result_does_not_change_recreated_source(session):
    old = session.commit_audio_input()
    session.register_history_item("audio", old.message)
    old_part = session.input_audio_part("audio")
    session.delete_history_item("audio")
    new = session.commit_audio_input()
    session.register_history_item("audio", new.message)
    before = session.history_snapshot()

    assert not session.update_input_transcript("audio", "stale", audio_part=old_part)
    assert session.history_snapshot() == before


def test_deleted_and_recreated_public_item_gets_new_source(session):
    session.register_history_item("reused", {"role": "user", "content": "same"})
    before = session.history_snapshot()

    assert session.delete_history_item("reused")
    deleted = session.history_snapshot()
    session.register_history_item("reused", {"role": "user", "content": "same"})
    recreated = session.history_snapshot()

    assert deleted.source_ids == ()
    assert deleted.history_revision > before.history_revision
    assert recreated.source_ids[0] > before.source_ids[0]
    assert recreated.source_revisions == (0,)


def test_rejected_deletion_does_not_change_history_revision(session):
    before = session.history_snapshot()

    assert not session.delete_history_item("absent")
    assert session.history_snapshot() == before


def test_epoch_change_alone_preserves_source_versions(session):
    session.register_history_item("user", {"role": "user", "content": "keep"})
    before = session.history_snapshot()

    session.barge_in()

    assert session.epoch == 1
    assert session.history_snapshot() == before


def test_live_session_resume_preserves_source_versions(session):
    session.register_history_item("user", {"role": "user", "content": "keep"})
    before = session.history_snapshot()
    session.detach_lease()

    assert session.resume_lease(expected_lease_generation=0) == 1
    assert session.history_snapshot() == before


def test_tighter_hard_cut_updates_revision_even_when_text_is_unchanged(session):
    response_id = session.begin_response()
    session.append_assistant_text("hello worlds")
    session.mark_audio_sent(duration_ms=2000, text_chars=12)
    session.acknowledge_playback(played_ms=2000)
    message = session.end_response()
    item_id = f"item_{response_id}"
    session.register_history_item(item_id, message)
    session.truncate_history_item(item_id, audio_end_ms=1999, hard=True)
    before = session.history_snapshot()

    session.truncate_history_item(item_id, audio_end_ms=1998, hard=True)
    after = session.history_snapshot()

    assert before.messages == after.messages == ({"role": "assistant", "content": "hello world"},)
    assert after.source_ids == before.source_ids
    assert after.source_revisions[0] > before.source_revisions[0]


def test_hard_truncate_without_owned_audio_is_a_revision_noop(session):
    session.register_history_item("item_imported", {"role": "assistant", "content": "keep"})
    before = session.history_snapshot()

    assert not session.truncate_history_item("item_imported", audio_end_ms=0, hard=True)
    assert session.history_snapshot() == before


def test_accepted_heard_prefix_updates_source_but_repeated_result_is_a_noop(session):
    session.replace_config(
        DuplexSessionConfig(model="test-model", playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value)
    )
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11)
    session.acknowledge_playback(played_ms=1000)
    message = session.end_response()
    session.register_history_item(f"item_{response_id}", message)
    before = session.history_snapshot()

    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    after = session.history_snapshot()

    assert before.messages == ({"role": "assistant", "content": "hello"},)
    assert after.messages == ({"role": "assistant", "content": "hell"},)
    assert after.source_ids == before.source_ids
    assert after.source_revisions[0] > before.source_revisions[0]
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    assert session.history_snapshot() == after


def test_same_visible_heard_prefix_with_new_cutoff_advances_revision(session):
    session.replace_config(
        DuplexSessionConfig(model="test-model", playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value)
    )
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11)
    session.acknowledge_playback(played_ms=1000)
    session.register_history_item(f"item_{response_id}", session.end_response())
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=900, char_end=4)
    before = session.history_snapshot()

    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    after = session.history_snapshot()

    assert after.messages == before.messages
    assert after.source_revisions[0] > before.source_revisions[0]


def test_repeated_truncate_is_a_revision_noop(session):
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11)
    session.acknowledge_playback(played_ms=2000)
    session.register_history_item(f"item_{response_id}", session.end_response())
    session.truncate_history_item(f"item_{response_id}", audio_end_ms=1000, hard=True)
    before = session.history_snapshot()

    session.truncate_history_item(f"item_{response_id}", audio_end_ms=1000, hard=True)

    assert session.history_snapshot() == before


def test_deleted_source_bookkeeping_is_released(session):
    session.register_history_item("removed", {"role": "user", "content": "private"})
    session.delete_history_item("removed")

    assert not session._conversation.sources


def test_snapshot_record_is_frozen_and_repeated_capture_does_not_revise(session):
    session.register_history_item("user", {"role": "user", "content": "keep"})
    before = session.history_snapshot()

    with pytest.raises(FrozenInstanceError):
        before.history_revision = 100

    assert session.history_snapshot() == before


def test_materialized_placeholder_keeps_its_original_position(session):
    session.reserve_history_item("reply")
    session.register_history_item("user", {"role": "user", "content": "next"})
    before = session.history_snapshot()

    session.register_history_item("reply", {"role": "assistant", "content": "earlier"})
    after = session.history_snapshot()

    assert after.item_ids == ("reply", "user")
    assert after.messages == (
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "next"},
    )
    assert after.source_ids[1] == before.source_ids[0]
    assert after.source_ids[0] > after.source_ids[1]
    assert after.source_revisions == (0, 0)


def test_truncation_after_snapshot_cleanup_updates_only_its_source(session):
    session.register_history_item("user", {"role": "user", "content": "keep"})
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11, audio_complete=True)
    session.acknowledge_playback(played_ms=2000)
    item_id = f"item_{response_id}"
    session.register_history_item(item_id, session.end_response())
    session.finalize_response_playback(response_id)
    before = session.history_snapshot()

    assert session.truncate_history_item(item_id, audio_end_ms=1000, hard=True)
    after = session.history_snapshot()

    assert before.messages[-1] == {"role": "assistant", "content": "hello world"}
    assert after.messages[-1] == {"role": "assistant", "content": "hello"}
    assert after.source_ids == before.source_ids
    assert after.source_revisions[0] == before.source_revisions[0]
    assert after.source_revisions[1] > before.source_revisions[1]


def test_repeated_hard_ack_after_finalization_is_a_revision_noop(session):
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11, audio_complete=True)
    session.register_history_item(f"item_{response_id}", session.end_response())

    apply_playback_ack(session, {"response_id": response_id, "played_ms": 1000, "truncate": True})
    before = session.history_snapshot()
    assert not session._conversation.hard_truncations_ms

    for cutoff in (1000, 1500):
        apply_playback_ack(session, {"response_id": response_id, "played_ms": cutoff, "truncate": True})
        assert session.history_snapshot() == before

    # Tightening evidence matters even when integer alignment keeps the text.
    apply_playback_ack(session, {"response_id": response_id, "played_ms": 999, "truncate": True})
    after = session.history_snapshot()
    assert after.messages == before.messages == ({"role": "assistant", "content": "hello"},)
    assert after.source_ids == before.source_ids
    assert after.source_revisions[0] > before.source_revisions[0]


def test_replacing_finalized_context_revises_removed_cutoff_metadata(session):
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11, audio_complete=True)
    item_id = f"item_{response_id}"
    session.register_history_item(item_id, session.end_response())
    apply_playback_ack(session, {"response_id": response_id, "played_ms": 1000, "truncate": True})
    before = session.history_snapshot()

    session.register_history_item(item_id, {"role": "assistant", "content": "hello"})
    after = session.history_snapshot()

    assert after.messages == before.messages
    assert after.source_ids == before.source_ids
    assert after.source_revisions[0] > before.source_revisions[0]


@pytest.mark.parametrize(
    ("transcript", "duration_ms", "calibrated", "revision_delta"),
    [
        pytest.param("hello world", 2000, False, 1, id="new-prefix"),
        pytest.param("hello world", 2000, True, 1, id="changed-prefix"),
        pytest.param("hello", 1000, True, 0, id="identical-prefix"),
    ],
)
def test_retiring_calibration_tracks_prefix_changes(session, transcript, duration_ms, calibrated, revision_delta):
    session.replace_config(
        DuplexSessionConfig(model="test-model", playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value)
    )
    session.register_history_item("user", {"role": "user", "content": "keep"})
    response_id = session.begin_response()
    session.append_assistant_text(transcript)
    session.mark_audio_sent(duration_ms=duration_ms, text_chars=len(transcript))
    session.acknowledge_playback(played_ms=1000)
    session.register_history_item(f"item_{response_id}", session.end_response())
    if calibrated:
        assert session.apply_heard_text(response_id, text=transcript, audio_end_ms=1000, char_end=5)
    before = session.history_snapshot()

    session.retire_history_calibration(response_id)
    after = session.history_snapshot()

    assert session.history_calibration_retired(response_id)
    assert (
        after.messages
        == before.messages
        == (
            {"role": "user", "content": "keep"},
            {"role": "assistant", "content": "hello"},
        )
    )
    assert after.source_ids == before.source_ids
    assert after.source_revisions == (before.source_revisions[0], before.source_revisions[1] + revision_delta)
    assert after.history_revision == before.history_revision + revision_delta

    session.retire_history_calibration(response_id)
    assert session.history_snapshot() == after


def test_retiring_calibration_clears_prefix_after_nontext_replacement(session):
    session.replace_config(
        DuplexSessionConfig(model="test-model", playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value)
    )
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(duration_ms=2000, text_chars=11)
    session.acknowledge_playback(played_ms=1000)
    item_id = f"item_{response_id}"
    session.register_history_item(item_id, session.end_response())
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=5)
    session.register_history_item(
        item_id,
        {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:replacement-audio"}}]},
    )
    before = session.history_snapshot()

    session.retire_history_calibration(response_id)

    assert not session.history_calibration_retired(response_id)
    assert session.history_snapshot() == before
