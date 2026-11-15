# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Memoized terms cannot outlive their session's active card nodes."""

import gc
import weakref
from dataclasses import replace

import pytest

from tests.engine.duplex.test_memory_card_auxiliary import _spoken
from tests.engine.duplex.test_memory_card_ledger import _admit, _operation
from tests.engine.duplex.test_memory_card_retrieval import CITY, _card, _retrieve
from tests.engine.duplex.test_memory_card_retrieval import session as _session_fixture
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextNodeRejectedError
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

session = _session_fixture
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def cached(session, node_id):
    return session.context_ledger.card_lexemes(node_id)


def test_empty_disabled_and_suppressed_queries_do_not_allocate_cache(session):
    node_id = _card(session, "city", "Tea")
    assert session.context_ledger._card_lexemes is None
    assert _retrieve(session, "") == ()
    assert _retrieve(session, "Tea", limit=0) == ()
    assert _retrieve(session, "Tea", suppress_all=True) == ()
    assert _retrieve(session, "Tea", keys=frozenset({CITY})) == ()
    assert session.context_ledger._card_lexemes is None
    assert [card.node_id for card in _retrieve(session, "Tea")] == [node_id]
    assert set(session.context_ledger._card_lexemes or {}) == {node_id}


def test_repeated_queries_reuse_immutable_terms_and_keep_literal_ranking(session):
    city = _card(session, "city", "The user lives in Hangzhou.")
    tea = _card(session, "tea", "Tea", key="user/self/drink")
    assert [card.node_id for card in _retrieve(session, "Hangzhou")] == [city]
    terms = cached(session, city)
    assert terms == frozenset({"current", "city", "lives", "hangzhou"})
    assert [card.node_id for card in _retrieve(session, "Tea")] == [tea]
    assert cached(session, city) is terms
    session.register_history_item("tail", {"role": "user", "content": "What time is it?"})
    assert [card.node_id for card in _retrieve(session, CITY)] == [city]
    assert cached(session, city) is terms


@pytest.mark.parametrize("change", ["edit", "delete", "retract", "replace"])
def test_retirement_releases_cached_terms_and_never_restores_old_head(session, change):
    old = _card(session, "city", "Hangzhou")
    _retrieve(session, "Hangzhou")
    terms = cached(session, old)
    reference = weakref.ref(terms)
    del terms
    if change == "edit":
        session.register_history_item("city", {"role": "user", "content": "Shanghai"})
    elif change == "delete":
        session.delete_history_item("city")
    elif change == "retract":
        assert _admit(session, _operation(session, "city", None, generation=1)).status == "applied"
    else:
        _card(session, "new", "Shanghai", generation=1)
    assert _retrieve(session, "Hangzhou") == ()
    gc.collect()
    assert reference() is None
    with pytest.raises(DuplexContextNodeRejectedError):
        cached(session, old)
    if change == "replace":
        assert [card.text for card in _retrieve(session, "Shanghai")] == ["Shanghai"]


def test_parent_invalidation_releases_cached_child_terms(session):
    parent = _card(session, "city", "Hangzhou")
    session.register_history_item("child", {"role": "user", "content": "Hangzhou detail"})
    child = _admit(
        session, _operation(session, "child", "Hangzhou detail", key="user/self/detail", parents=(parent,))
    ).node_ids[0]
    _retrieve(session, "Hangzhou")
    references = [weakref.ref(cached(session, node_id)) for node_id in (parent, child)]
    session.delete_history_item("city")
    assert _retrieve(session, "Hangzhou") == ()
    gc.collect()
    assert all(ref() is None for ref in references)


def test_heard_prefix_revision_releases_terms_even_when_visible_text_is_identical(session):
    response_id = _spoken(session)
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=900, char_end=4)
    node_id = _admit(session, _operation(session, f"item_{response_id}", "hell")).node_ids[0]
    _retrieve(session, "hell")
    reference = weakref.ref(cached(session, node_id))
    assert session.apply_heard_text(response_id, text="hello world", audio_end_ms=1000, char_end=4)
    assert _retrieve(session, "hell") == ()
    gc.collect()
    assert reference() is None


def test_failed_batch_preserves_valid_cached_terms(session):
    node_id = _card(session, "city", "Hangzhou")
    _retrieve(session, "Hangzhou")
    terms = cached(session, node_id)
    operation = _operation(session, "city", "Other", generation=1)
    stale = replace(operation, evidence=(replace(operation.evidence[0], source_id=9999),))
    assert _admit(session, stale).status == "stale"
    assert [card.node_id for card in _retrieve(session, "Hangzhou")] == [node_id]
    assert cached(session, node_id) is terms


def test_identical_node_ids_and_text_in_other_session_never_share_cache(session):
    node_id = _card(session, "city", "Tea")
    other = DuplexEngineSession(session_id=session.session_id, config=DuplexSessionConfig(model="test-model"))
    assert _card(other, "city", "Tea") == node_id
    _retrieve(session, "Tea")
    _retrieve(other, "Tea")
    assert cached(session, node_id) == cached(other, node_id)
    assert cached(session, node_id) is not cached(other, node_id)


def test_live_resume_keeps_cache_but_close_releases_and_disables_it(session):
    node_id = _card(session, "city", "Tea")
    _retrieve(session, "Tea")
    terms = cached(session, node_id)
    session.detach_lease()
    session.resume_lease(expected_lease_generation=session.lease_generation)
    assert cached(session, node_id) is terms
    reference = weakref.ref(terms)
    del terms
    session.begin_close(reason="test")
    gc.collect()
    assert reference() is None
    assert session.capture_memory_prompt(query="Tea").cards == ()
    diagnostic = cached(session, node_id)
    assert diagnostic == frozenset({"current", "city", "tea"})
    assert cached(session, node_id) is not diagnostic


def test_cache_does_not_keep_retired_terms_during_capacity_churn(session):
    session.register_history_item("facts", {"role": "user", "content": "A collection of facts"})
    for start in range(0, 256, 8):
        assert (
            _admit(
                session,
                *[_operation(session, "facts", "Tea", key=f"user/self/fact{i}") for i in range(start, start + 8)],
            ).status
            == "applied"
        )
    assert len(_retrieve(session, "Tea")) == 16
    references = {
        head.fact_key: weakref.ref(cached(session, head.node_id)) for head in session.context_ledger.card_catalog()
    }
    for index in range(20):
        head = next(h for h in session.context_ledger.card_catalog() if h.fact_key == "user/self/fact0")
        assert (
            _admit(
                session, _operation(session, "facts", f"Tea {index}", key=head.fact_key, generation=head.generation)
            ).status
            == "applied"
        )
        assert len(_retrieve(session, "Tea")) == 16
    gc.collect()
    assert references["user/self/fact0"]() is None
    assert all(ref() is not None for key, ref in references.items() if key != "user/self/fact0")
    session.delete_history_item("facts")
    assert _retrieve(session, "Tea") == ()
    gc.collect()
    assert all(ref() is None for ref in references.values())


def test_large_cjk_cards_fall_back_at_retained_term_budget_and_reclaim_it(session):
    nodes = []
    for index in range(80):
        text = "".join(chr(0x4E00 + (index * 521 + offset) % 20000) for offset in range(512))
        nodes.append(_card(session, f"source{index}", text, key=f"user/self/fact{index}"))
    for node_id in nodes:
        cached(session, node_id)
    ledger = session.context_ledger
    assert ledger._card_lexeme_bytes <= 4 * 1024 * 1024
    assert 0 < len(ledger._card_lexemes) < len(nodes)
    retained_id = next(iter(ledger._card_lexemes))
    overflow_id = next(node_id for node_id in nodes if node_id not in ledger._card_lexemes)
    terms = cached(session, overflow_id)
    assert cached(session, overflow_id) == terms
    assert cached(session, overflow_id) is not terms
    assert [card.node_id for card in _retrieve(session, "fact79")] == [nodes[79]]
    before = ledger._card_lexeme_bytes
    source = f"source{nodes.index(retained_id)}"
    session.delete_history_item(source)
    ledger.refresh(session.history_snapshot())
    assert ledger._card_lexeme_bytes < before
    assert cached(session, overflow_id) == terms
    assert overflow_id in ledger._card_lexemes
    session.begin_close(reason="budget-test")
    assert ledger._card_lexeme_bytes == 0
    assert ledger._card_lexemes is None
