# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A prompt selection remains owned and revalidated by its originating session."""

from dataclasses import replace

import pytest

from tests.engine.duplex.test_memory_card_retrieval import CITY, _block, _card
from tests.engine.duplex.test_memory_card_retrieval import session as _session_fixture
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

session = _session_fixture
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def capture(session):
    return session.capture_memory_prompt(query="Which city is Hangzhou?")


def test_current_receipt_selects_only_captured_nodes_and_never_writes_history(session):
    node_id = _card(session, "city", "I live in Hangzhou.")
    before = session.history_snapshot()
    receipt = capture(session)
    assert [card.node_id for card in receipt.cards] == [node_id]
    assert session.memory_prompt_matches(receipt, (node_id,))
    assert session.memory_prompt_matches(receipt, ())
    assert session.history_snapshot() == before


@pytest.mark.parametrize("change", ["edit", "delete", "supersede", "suppress", "catchup", "epoch", "config", "close"])
def test_selection_is_rejected_when_evidence_or_eligibility_changes(session, change):
    node_id = _card(session, "city", "I live in Hangzhou.")
    receipt = capture(session)
    if change == "edit":
        session.register_history_item("city", {"role": "user", "content": "I live in Shanghai."})
    elif change == "delete":
        session.delete_history_item("city")
    elif change == "supersede":
        _card(session, "correction", "I live in Shanghai.", generation=1)
    elif change == "suppress":
        _block(session, session.memory_suppression, "city", keys=frozenset({CITY}))
    elif change == "catchup":
        session.restart_memory_reconciliation()
    elif change == "epoch":
        session.epoch += 1
    elif change == "config":
        session.config_revision += 1
    else:
        session.begin_close(reason="test")
    assert not session.memory_prompt_matches(receipt, (node_id,))


def test_reconciliation_blocks_retrieval_before_first_extraction_capture(session):
    _card(session, "city", "I live in Hangzhou.")
    session.restart_memory_reconciliation()
    assert capture(session).cards == ()


def test_receipt_cannot_cross_sessions_with_same_public_ids_and_fact_keys(session):
    node_id = _card(session, "city", "I live in Hangzhou.")
    receipt = capture(session)
    other = DuplexEngineSession(session_id=session.session_id, config=DuplexSessionConfig(model="test-model"))
    assert _card(other, "city", "I live in Hangzhou.") == node_id
    assert not other.memory_prompt_matches(receipt, (node_id,))


@pytest.mark.parametrize("ids", [[0], (True,), (-1,), (9999,), (0, 0), "0", None])
def test_invalid_or_foreign_selections_fail_closed(session, ids):
    _card(session, "city", "I live in Hangzhou.")
    assert not session.memory_prompt_matches(capture(session), ids)


def test_forged_card_body_in_a_copied_receipt_is_rejected(session):
    node_id = _card(session, "city", "I live in Hangzhou.")
    receipt = capture(session)
    forged = replace(receipt, cards=(replace(receipt.cards[0], text="Forged content"),))
    assert not session.memory_prompt_matches(forged, (node_id,))


def test_ordinary_tail_append_does_not_invalidate_an_unchanged_card(session):
    node_id = _card(session, "city", "I live in Hangzhou.")
    receipt = capture(session)
    session.register_history_item("tail", {"role": "user", "content": "What is the time?"})
    assert session.memory_prompt_matches(receipt, (node_id,))


@pytest.mark.parametrize("omitted", [[0], (True,), (-1,), (9999,), (0, 0), "0", None])
def test_omitted_source_references_are_validated_on_the_owner(session, omitted):
    node_id = _card(session, "city", "I live in Hangzhou.")
    assert not session.memory_prompt_matches(capture(session), (node_id,), omitted_source_ids=omitted)


def test_no_background_coverage_is_not_a_freshness_certificate(session):
    node_id = _card(session, "city", "I live in Hangzhou.")
    receipt = capture(session)
    sources = session.history_snapshot().source_ids
    assert not session.memory_prompt_matches(receipt, (node_id,), omitted_source_ids=sources)
    # The existing no-card fallback remains usable with that same base.
    assert session.memory_prompt_matches(receipt, (), omitted_source_ids=sources)


def test_changed_heard_cutoff_rejects_same_text_selection(session):
    from tests.engine.duplex.test_memory_card_auxiliary import _spoken
    from tests.engine.duplex.test_memory_card_ledger import _admit, _operation

    response_id = _spoken(session)
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=900, char_end=4)
    node_id = _admit(session, _operation(session, f"item_{response_id}", "hell")).node_ids[0]
    receipt = session.capture_memory_prompt(query="hell")
    assert session.memory_prompt_matches(receipt, (node_id,))
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    assert not session.memory_prompt_matches(receipt, (node_id,))


def test_invalidated_parent_rejects_child_even_with_current_direct_source(session):
    from tests.engine.duplex.test_memory_card_ledger import _admit, _operation

    parent_id = _card(session, "city", "I live in Hangzhou.")
    session.register_history_item("detail", {"role": "user", "content": "A detail about Hangzhou."})
    child_id = _admit(
        session,
        _operation(session, "detail", "Hangzhou detail", key="user/self/detail", parents=(parent_id,)),
    ).node_ids[0]
    receipt = capture(session)
    assert session.memory_prompt_matches(receipt, (child_id,))
    session.delete_history_item("city")
    assert not session.memory_prompt_matches(receipt, (child_id,))
