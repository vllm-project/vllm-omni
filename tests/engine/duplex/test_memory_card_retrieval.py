# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Selection must never bypass source validity or a pending correction."""

from dataclasses import replace

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.memory_types import (
    DuplexMemoryCardBatch,
    DuplexMemoryCardOperation,
    DuplexMemoryEvidenceSpan,
)
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CITY = "user/self/current_city"
DRINK = "user/self/drink"


@pytest.fixture
def session() -> DuplexEngineSession:
    return DuplexEngineSession(session_id="memory-retrieval", config=DuplexSessionConfig(model="test-model"))


def _evidence(session, item_id):
    snapshot = session.history_snapshot()
    index = snapshot.item_ids.index(item_id)
    return DuplexMemoryEvidenceSpan(
        snapshot.source_ids[index], snapshot.source_revisions[index], 0, len(snapshot.messages[index]["content"])
    )


def _card(session, item_id, text, *, key=CITY, generation=0):
    session.register_history_item(item_id, {"role": "user", "content": text})
    result = session.admit_memory_cards(
        DuplexMemoryCardBatch(
            operations=(DuplexMemoryCardOperation(key, generation, (_evidence(session, item_id),), text),),
            catalog_revision=session.context_ledger.card_catalog_revision,
        )
    )
    assert result.status == "applied"
    return result.node_ids[0]


def _retrieve(session, query, *, keys=frozenset(), suppress_all=False, limit=16):
    from vllm_omni.engine.duplex.session.memory_retrieval import retrieve_memory_cards

    return retrieve_memory_cards(
        snapshot=session.history_snapshot(),
        ledger=session.context_ledger,
        query=query,
        suppressed_keys=keys,
        suppress_all=suppress_all,
        max_candidates=limit,
    )


def _suppression(session):
    from vllm_omni.engine.duplex.session.memory_retrieval import DuplexMemorySuppression

    assert isinstance(session.memory_suppression, DuplexMemorySuppression)
    return session.memory_suppression


def _block(session, state, item_id, *, keys=None):
    return state.block(
        snapshot=session.history_snapshot(),
        evidence=(_evidence(session, item_id),),
        keys=keys,
        catalog=session.context_ledger.card_catalog(),
    )


def test_english_query_matches_fact_key_and_text(session):
    city = _card(session, "city", "The user lives in Hangzhou.")
    _card(session, "drink", "The user drinks tea.", key=DRINK)

    assert [card.node_id for card in _retrieve(session, "Which city do I live in?")] == [city]


def test_chinese_query_uses_character_bigrams(session):
    city = _card(session, "city", "用户目前住在杭州。")
    _card(session, "drink", "用户喜欢喝茶。", key=DRINK)

    assert [card.node_id for card in _retrieve(session, "我现在住在哪个城市？")] == [city]


def test_explicit_key_match_precedes_more_recent_text_match(session):
    city = _card(session, "city", "Hangzhou")
    _card(session, "drink", "The user visited a city.", key=DRINK)

    assert _retrieve(session, CITY)[0].node_id == city


def test_equal_scores_prefer_newer_node_and_are_repeatable(session):
    first = _card(session, "a", "Tea", key="user/self/favorite_drink")
    second = _card(session, "b", "Tea", key="user/self/drink")

    for _ in range(3):
        assert [card.node_id for card in _retrieve(session, "Tea")] == [second, first]


@pytest.mark.parametrize("query", ["", "What is the weather?", "x" * 4097])
def test_empty_unrelated_or_oversized_query_returns_no_cards(session, query):
    _card(session, "city", "The user lives in Hangzhou.")

    assert _retrieve(session, query) == ()


def test_source_edit_is_checked_before_retrieval(session):
    _card(session, "city", "Hangzhou")
    session.register_history_item("city", {"role": "user", "content": "Shanghai"})

    assert _retrieve(session, "Hangzhou") == ()
    assert session.context_ledger.card_catalog()[0].node_id is None


def test_superseded_card_does_not_match_old_value(session):
    _card(session, "a", "Shanghai")
    _card(session, "b", "Hangzhou", generation=1)

    assert _retrieve(session, "Shanghai") == ()
    assert [card.text for card in _retrieve(session, "Hangzhou")] == ["Hangzhou"]


def test_suppression_excludes_only_named_keys(session):
    _card(session, "city", "Hangzhou")
    tea = _card(session, "drink", "Tea", key=DRINK)

    assert [card.node_id for card in _retrieve(session, "Hangzhou Tea", keys=frozenset({CITY}))] == [tea]
    assert _retrieve(session, "Hangzhou Tea", suppress_all=True) == ()


def test_candidate_limit_is_enforced_without_rewriting_cards(session):
    for index in range(20):
        _card(session, str(index), "Tea", key=f"user/self/fact{index}")
    before = session.history_snapshot()

    assert [card.node_id for card in _retrieve(session, "Tea", limit=2)] == [20, 19]
    assert _retrieve(session, "Tea", limit=0) == ()
    assert len(session.context_ledger.card_candidates()) == 20
    assert session.history_snapshot() == before


def test_unknown_key_conflict_stays_blocked_on_next_query(session):
    _card(session, "city", "Shanghai")
    session.register_history_item("correction", {"role": "user", "content": "That is no longer true."})
    state = _suppression(session)

    barrier = _block(session, state, "correction")

    assert barrier is not None
    assert state.suppress_all
    assert _retrieve(session, "Shanghai", suppress_all=state.suppress_all) == ()
    session.register_history_item("query", {"role": "user", "content": "Where do I live?"})
    assert state.suppress_all
    assert _retrieve(session, "city", suppress_all=state.suppress_all) == ()


def test_empty_card_admission_does_not_resolve_barrier(session):
    _card(session, "city", "Shanghai")
    state = _suppression(session)
    _block(session, state, "city", keys=frozenset({CITY}))

    session.admit_memory_cards(DuplexMemoryCardBatch((), session.context_ledger.card_catalog_revision))

    assert state.suppressed_keys == frozenset({CITY})


def test_explicit_resolution_requires_current_trigger_revision(session):
    _card(session, "city", "Shanghai")
    state = _suppression(session)
    barrier = _block(session, state, "city", keys=frozenset({CITY}))
    session.register_history_item("city", {"role": "user", "content": "Hangzhou"})

    accepted = state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=barrier.evidence,
        keys=frozenset({CITY}),
    )

    assert not accepted
    assert state.suppressed_keys == frozenset({CITY})


def test_partial_catalog_reconciliation_cannot_clear_global_barrier(session):
    _card(session, "city", "Shanghai")
    _card(session, "drink", "Tea", key=DRINK)
    state = _suppression(session)
    barrier = _block(session, state, "city")
    snapshot = session.history_snapshot()

    assert state.resolve(
        snapshot=snapshot,
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=barrier.evidence,
        keys=frozenset({CITY}),
    )
    assert state.suppress_all
    assert state.resolve(
        snapshot=snapshot,
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=barrier.evidence,
        keys=frozenset({DRINK}),
    )
    assert not state.suppress_all


def test_partial_trigger_coverage_does_not_clear_barrier(session):
    _card(session, "city", "Shanghai")
    state = _suppression(session)
    barrier = _block(session, state, "city", keys=frozenset({CITY}))
    span = barrier.evidence[0]

    assert state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=(replace(span, char_end=3),),
        keys=frozenset({CITY}),
    )
    assert state.suppressed_keys == frozenset({CITY})
    assert state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=(replace(span, char_start=3),),
        keys=frozenset({CITY}),
    )
    assert state.suppressed_keys == frozenset()


def test_unknown_or_stale_resolution_cannot_clear_new_barrier(session):
    _card(session, "city", "Shanghai")
    state = _suppression(session)
    barrier = _block(session, state, "city", keys=frozenset({CITY}))
    version = state.version

    assert not state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation + 1,
        evidence=barrier.evidence,
        keys=frozenset({CITY}),
    )
    assert not state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=999,
        generation=barrier.generation,
        evidence=barrier.evidence,
        keys=frozenset({CITY}),
    )
    assert state.version == version
    assert state.suppressed_keys == frozenset({CITY})


def test_new_head_generation_invalidates_partial_resolution(session):
    _card(session, "city", "Shanghai")
    _card(session, "drink", "Tea", key=DRINK)
    session.register_history_item("change", {"role": "user", "content": "Some earlier facts have changed."})
    state = _suppression(session)
    barrier = _block(session, state, "change")

    assert state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=barrier.evidence,
        keys=frozenset({CITY}),
    )
    _card(session, "new-city", "Hangzhou", generation=1)
    assert state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=barrier.evidence,
        keys=frozenset({DRINK}),
    )

    assert state.suppress_all
    assert state.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=barrier.barrier_id,
        generation=barrier.generation,
        evidence=barrier.evidence,
        keys=frozenset({CITY}),
    )
    assert not state.suppress_all


def test_retrieval_builds_source_revision_map_once(session):
    from vllm_omni.engine.duplex.session.memory_retrieval import retrieve_memory_cards

    for index in range(20):
        _card(session, str(index), "Tea", key=f"user/self/fact{index}")
    iterations = []

    class CountingSources(tuple):
        def __iter__(self):
            iterations.append(len(self))
            return super().__iter__()

    snapshot = session.history_snapshot()
    counted = replace(snapshot, source_ids=CountingSources(snapshot.source_ids))

    cards = retrieve_memory_cards(
        snapshot=counted,
        ledger=session.context_ledger,
        query="Tea",
        suppressed_keys=frozenset(),
        suppress_all=False,
        max_candidates=16,
    )

    assert len(cards) == 16
    assert iterations == [20]
