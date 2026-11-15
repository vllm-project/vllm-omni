# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit retirement must preserve every index over the derived graph."""

import gc
import weakref

import pytest

from tests.engine.duplex.test_memory_card_ledger import _admit, _head, _operation
from tests.engine.duplex.test_memory_card_ledger import session as _session_fixture
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import (
    DuplexContextLedger,
    DuplexContextLevel,
    DuplexContextNodeRejectedError,
    source_refs,
)
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

session = _session_fixture
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source(session, item_id, text):
    session.register_history_item(item_id, {"role": "user", "content": text})


def _summary(session, item_id, *, parents=()):
    snapshot = session.history_snapshot()
    index = snapshot.item_ids.index(item_id)
    return session.context_ledger.add_node(
        snapshot,
        level=DuplexContextLevel.SUMMARY,
        sources=source_refs(snapshot, snapshot.source_ids[index]),
        parents=parents,
        message={"role": "system", "content": "Historical summary"},
    )


def _card(session, item_id, *, key, parents=(), generation=0):
    result = _admit(
        session,
        _operation(session, item_id, "Tea", key=key, parents=parents, generation=generation),
    )
    assert result.status == "applied"
    return result.node_ids[0]


def test_discard_retires_mixed_descendants_in_insertion_order(session):
    for item_id in ("root", "l1", "card", "unrelated", "child"):
        _source(session, item_id, "Tea")
    ledger = session.context_ledger
    root = _summary(session, "root")
    snapshot = session.history_snapshot()
    l1_source = snapshot.source_ids[snapshot.item_ids.index("l1")]
    (l1,) = ledger.add_l1_nodes(snapshot, messages={l1_source: {"role": "user", "content": "Tea"}})
    first = _card(session, "card", key="user/self/drink", parents=(root.node_id,))
    unrelated = _card(session, "unrelated", key="user/self/unrelated")
    child = _card(session, "child", key="user/self/detail", parents=(first,))
    before_catalog = ledger.card_catalog_revision

    assert ledger.discard(root.node_id) == (1, 3, 5)

    assert (root.node_id, l1.node_id, first, unrelated, child) == (1, 2, 3, 4, 5)
    assert tuple(node.node_id for node in ledger.nodes()) == (2, 4)
    assert tuple(node.node_id for node in ledger.card_candidates()) == (4,)
    assert ledger.l1_node_for(l1_source) is l1
    assert ledger.card_catalog_revision == before_catalog + 1
    for key in ("user/self/drink", "user/self/detail"):
        head = _head(session, key)
        assert (head.node_id, head.generation) == (None, 2)
    assert session.history_snapshot() == snapshot
    assert ledger.refresh(snapshot) == ()


def test_discard_releases_descendant_terms_and_preserves_unrelated_cache(session):
    for item_id in ("root", "card", "child", "unrelated"):
        _source(session, item_id, "Tea")
    ledger = session.context_ledger
    root = _summary(session, "root")
    first = _card(session, "card", key="user/self/drink", parents=(root.node_id,))
    child = _card(session, "child", key="user/self/detail", parents=(first,))
    unrelated = _card(session, "unrelated", key="user/self/unrelated")
    references = [weakref.ref(ledger.card_lexemes(node_id)) for node_id in (first, child)]
    survivor = ledger.card_lexemes(unrelated)
    before_bytes = ledger._card_lexeme_bytes

    assert ledger.discard(root.node_id) == (root.node_id, first, child)

    gc.collect()
    assert all(reference() is None for reference in references)
    assert ledger.card_lexemes(unrelated) is survivor
    assert 0 < ledger._card_lexeme_bytes < before_bytes
    for node_id in (first, child):
        with pytest.raises(DuplexContextNodeRejectedError, match="memory_card_not_active"):
            ledger.card_lexemes(node_id)


def test_discard_last_cached_card_reclaims_terms_and_keeps_tombstone(session):
    _source(session, "card", "Tea")
    ledger = session.context_ledger
    node_id = _card(session, "card", key="user/self/drink")
    reference = weakref.ref(ledger.card_lexemes(node_id))

    assert ledger.discard(node_id) == (node_id,)

    gc.collect()
    assert reference() is None
    assert ledger._card_lexeme_bytes == 0
    assert ledger._card_lexemes is None
    assert ledger.card_candidates() == ()
    head = _head(session, "user/self/drink")
    assert (head.node_id, head.generation) == (None, 2)
    assert _admit(session, _operation(session, "card", "Tea", key=head.fact_key)).status == "stale"
    assert ledger.card_candidates() == ()


def test_discard_unknown_and_repeated_are_noops_without_consuming_ids(session):
    _source(session, "card", "Tea")
    ledger = session.context_ledger
    node_id = _card(session, "card", key="user/self/drink")
    terms = ledger.card_lexemes(node_id)
    before = (ledger.nodes(), ledger.card_catalog(), ledger.card_catalog_revision)
    snapshot = session.history_snapshot()

    assert ledger.discard(9999) == ()
    assert (ledger.nodes(), ledger.card_catalog(), ledger.card_catalog_revision) == before
    assert ledger.card_lexemes(node_id) is terms
    assert ledger.discard(node_id) == (node_id,)
    revision = ledger.card_catalog_revision
    assert ledger.discard(node_id) == ()
    assert ledger.card_catalog_revision == revision
    assert session.history_snapshot() == snapshot
    replacement = _card(session, "card", key="user/self/drink", generation=2)
    assert replacement == node_id + 1


def test_discard_old_l1_preserves_newer_lookup_and_clears_selected_l1(session):
    _source(session, "source", "Before")
    ledger = session.context_ledger
    old = session.history_snapshot()
    source_id = old.source_ids[0]
    (first,) = ledger.add_l1_nodes(old, messages={source_id: {"role": "user", "content": "Before"}})
    _source(session, "source", "After")
    current = session.history_snapshot()
    (second,) = ledger.add_l1_nodes(current, messages={source_id: {"role": "user", "content": "After"}})
    revision = ledger.card_catalog_revision

    assert ledger.discard(first.node_id) == (first.node_id,)
    assert ledger.l1_node_for(source_id) is second
    assert ledger.discard(second.node_id) == (second.node_id,)
    assert ledger.l1_node_for(source_id) is None
    assert ledger.card_catalog_revision == revision
    assert session.history_snapshot() == current
    assert _summary(session, "source").node_id == 3


def test_discard_only_summaries_does_not_change_card_catalog(session):
    _source(session, "root", "Tea")
    _source(session, "card", "Tea")
    ledger = session.context_ledger
    root = _summary(session, "root")
    child = _summary(session, "root", parents=(root.node_id,))
    card = _card(session, "card", key="user/self/drink")
    revision = ledger.card_catalog_revision

    assert ledger.discard(root.node_id) == (root.node_id, child.node_id)

    assert ledger.card_catalog_revision == revision
    assert tuple(node.node_id for node in ledger.card_candidates()) == (card,)


def test_rejected_summary_replacement_preserves_old_node_and_dependents(session):
    _source(session, "root", "Tea")
    _source(session, "card", "Tea")
    ledger = session.context_ledger
    old = _summary(session, "root")
    card = _card(session, "card", key="user/self/drink", parents=(old.node_id,))
    snapshot = session.history_snapshot()
    _source(session, "root", "Edited")

    with pytest.raises(DuplexContextNodeRejectedError, match="stale_source"):
        ledger.add_node(
            session.history_snapshot(),
            level=DuplexContextLevel.SUMMARY,
            sources=source_refs(snapshot, snapshot.source_ids[0]),
            message={"role": "system", "content": "Replacement"},
        )

    # Rejection itself does not mutate the graph. A later refresh prunes stale nodes.
    assert tuple(node.node_id for node in ledger.nodes()) == (old.node_id, card)


def test_flat_source_summary_replacement_survives_discard_of_old_summary(session):
    _source(session, "root", "Tea")
    _source(session, "new", "A later fact")
    _source(session, "card", "Tea")
    ledger = session.context_ledger
    old = _summary(session, "root")
    card = _card(session, "card", key="user/self/drink", parents=(old.node_id,))
    replacement = _summary(session, "new")

    assert ledger.discard(old.node_id) == (old.node_id, card)

    assert ledger.valid(replacement.node_id)
    assert replacement.parents == ()
    assert ledger.refresh(session.history_snapshot()) == ()
    assert ledger.nodes() == (replacement,)


def test_discard_invalidates_descendants_without_touching_history_or_other_nodes():
    session = DuplexEngineSession(session_id="discard", config=DuplexSessionConfig(model="test"))
    session.register_history_item("one", {"role": "user", "content": "one"})
    session.register_history_item("two", {"role": "user", "content": "two"})
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    old = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.MEDIA_DROPPED,
        sources=source_refs(snapshot, snapshot.source_ids[0]),
        message=snapshot.messages[0],
    )
    child = ledger.add_node(
        snapshot, level=DuplexContextLevel.SUMMARY, sources=(), parents=(old.node_id,), message={"content": "child"}
    )
    grandchild = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.SUMMARY,
        sources=(),
        parents=(child.node_id,),
        message={"content": "grandchild"},
    )
    independent = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.SUMMARY,
        sources=source_refs(snapshot, *snapshot.source_ids),
        message={"content": "replacement"},
    )
    assert ledger.discard(old.node_id) == (old.node_id, child.node_id, grandchild.node_id)
    assert ledger.nodes() == (independent,)
    assert ledger.l1_node_for(snapshot.source_ids[0]) is None
    assert ledger.discard(old.node_id) == ()
    assert ledger.discard(99999) == ()
    assert session.history_snapshot() == snapshot
    replacement = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.MEDIA_DROPPED,
        sources=source_refs(snapshot, snapshot.source_ids[0]),
        message=snapshot.messages[0],
    )
    assert replacement.node_id > independent.node_id
    assert ledger.l1_node_for(snapshot.source_ids[0]) == replacement
    assert ledger.refresh(snapshot) == ()
