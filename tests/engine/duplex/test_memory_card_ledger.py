# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Memory cards must never replace newer evidence or revive superseded facts."""

from dataclasses import replace

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import (
    DuplexContextLevel,
    DuplexContextNodeRejectedError,
    source_refs,
)
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CITY = "user/self/current_city"


@pytest.fixture
def session() -> DuplexEngineSession:
    return DuplexEngineSession(session_id="memory-cards", config=DuplexSessionConfig(model="test-model"))


def _user(text: str) -> dict[str, object]:
    return {"role": "user", "content": text}


def _operation(session, item_id, text, *, key=CITY, generation=0, parents=()):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryCardOperation, DuplexMemoryEvidenceSpan

    snapshot = session.history_snapshot()
    index = snapshot.item_ids.index(item_id)
    content = snapshot.messages[index]["content"]
    assert isinstance(content, str)
    return DuplexMemoryCardOperation(
        fact_key=key,
        expected_generation=generation,
        evidence=(
            DuplexMemoryEvidenceSpan(
                source_id=snapshot.source_ids[index],
                revision=snapshot.source_revisions[index],
                char_start=0,
                char_end=len(content),
            ),
        ),
        text=text,
        parents=parents,
    )


def _admit(session, *operations, catalog_revision=None):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryCardBatch

    ledger = session.context_ledger
    return session.admit_memory_cards(
        DuplexMemoryCardBatch(
            operations=operations,
            catalog_revision=ledger.card_catalog_revision if catalog_revision is None else catalog_revision,
        )
    )


def _head(session, key=CITY):
    return next(head for head in session.context_ledger.card_catalog() if head.fact_key == key)


def test_card_admission_preserves_authoritative_history(session):
    session.register_history_item("a", _user("I live in Shanghai."))
    before = session.history_snapshot()

    result = _admit(session, _operation(session, "a", "The user lives in Shanghai."))

    assert result.status == "applied"
    assert result.node_ids == (1,)
    (node,) = session.context_ledger.card_candidates()
    assert node.level is DuplexContextLevel.MEMORY_CARD
    assert node.message["content"] == "The user lives in Shanghai."
    assert node.card.fact_key == CITY
    assert node.card.generation == 1
    assert session.history_snapshot() == before


def test_late_invalid_candidate_is_atomic(session):
    session.register_history_item("a", _user("I live in Shanghai."))
    valid = _operation(session, "a", "Shanghai")
    invalid = replace(valid, fact_key="user/self/food", evidence=(replace(valid.evidence[0], source_id=999),))

    result = _admit(session, valid, invalid)

    assert result.status == "stale"
    assert session.context_ledger.nodes() == ()
    assert session.context_ledger.card_catalog() == ()
    assert _admit(session, valid).node_ids == (1,)


def test_new_source_correction_supersedes_unchanged_source(session):
    session.register_history_item("a", _user("I live in Shanghai."))
    _admit(session, _operation(session, "a", "Shanghai"))
    old = session.history_snapshot()
    session.register_history_item("b", _user("Actually, I live in Hangzhou."))

    result = _admit(session, _operation(session, "b", "Hangzhou", generation=1))

    assert result.status == "applied"
    assert _head(session).generation == 2
    assert _head(session).node_id == 2
    assert not session.context_ledger.valid(1)
    assert [node.message["content"] for node in session.context_ledger.card_candidates()] == ["Hangzhou"]
    assert session.history_snapshot().source_revisions[0] == old.source_revisions[0]


def test_replacement_retires_old_dependents_but_preserves_unrelated_branch(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    ledger = session.context_ledger
    dependent = ledger.add_node(
        session.history_snapshot(),
        level=DuplexContextLevel.SUMMARY,
        sources=(),
        parents=(1,),
        message={"role": "user", "content": "Old-city explanation"},
    )
    _admit(session, _operation(session, "a", "Tea", key="user/self/drink"))
    session.register_history_item("b", _user("Hangzhou"))

    result = _admit(session, _operation(session, "b", "Hangzhou", generation=1))

    assert result.status == "applied"
    assert not ledger.valid(dependent.node_id)
    assert _head(session, "user/self/drink").node_id == 3
    assert ledger.valid(3)


def test_supersedes_is_not_a_parent(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("b", _user("Hangzhou"))
    _admit(session, _operation(session, "b", "Hangzhou", generation=1))
    session.register_history_item("a", _user("An edited old message"))

    session.context_ledger.refresh(session.history_snapshot())

    (node,) = session.context_ledger.card_candidates()
    assert node.node_id == 2
    assert node.parents == ()
    assert node.card.supersedes_node_id == 1
    assert node.message["content"] == "Hangzhou"


def test_replacement_invalidation_never_revives_old_card(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("b", _user("Hangzhou"))
    _admit(session, _operation(session, "b", "Hangzhou", generation=1))
    session.delete_history_item("b")

    session.context_ledger.refresh(session.history_snapshot())

    assert session.context_ledger.card_candidates() == ()
    assert _head(session).node_id is None
    assert _head(session).generation == 3
    result = _admit(session, _operation(session, "a", "Shanghai", generation=3))
    assert result.status == "stale"
    assert session.context_ledger.card_candidates() == ()


def test_stale_generation_cannot_overwrite_new_head(session):
    session.register_history_item("a", _user("Shanghai"))
    old_candidate = _operation(session, "a", "Shanghai")
    _admit(session, old_candidate)
    session.register_history_item("b", _user("Hangzhou"))
    _admit(session, _operation(session, "b", "Hangzhou", generation=1))

    result = _admit(session, old_candidate)

    assert result.status == "stale"
    assert _head(session).node_id == 2


def test_fresh_extraction_of_older_evidence_cannot_replace_later_fact(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("b", _user("Hangzhou"))
    _admit(session, _operation(session, "b", "Hangzhou", generation=1))
    session.register_history_item("a", _user("Shanghai, edited after the correction."))

    result = _admit(session, _operation(session, "a", "Shanghai", generation=2))

    assert result.status == "stale"
    assert _head(session).node_id == 2


def test_old_public_item_id_does_not_restore_source(session):
    session.register_history_item("a", _user("Shanghai"))
    old_candidate = _operation(session, "a", "Shanghai")
    session.delete_history_item("a")
    session.register_history_item("a", _user("Hangzhou"))

    result = _admit(session, old_candidate)

    assert result.status == "stale"
    assert session.context_ledger.nodes() == ()
    assert _admit(session, _operation(session, "a", "Hangzhou")).node_ids == (1,)


def test_placeholder_keeps_original_slot_order(session):
    session.reserve_history_item("assistant-a")
    session.register_history_item("user-b", _user("Hangzhou"))
    _admit(session, _operation(session, "user-b", "Hangzhou"))
    session.register_history_item("assistant-a", {"role": "assistant", "content": "Shanghai"})
    snapshot = session.history_snapshot()
    assert snapshot.source_ids[0] > snapshot.source_ids[1]

    result = _admit(session, _operation(session, "assistant-a", "Shanghai", generation=1))

    assert result.status == "stale"
    assert [node.message["content"] for node in session.context_ledger.card_candidates()] == ["Hangzhou"]


def test_source_edit_invalidates_card_and_versions_tombstone_once(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("a", _user("Hangzhou"))
    ledger = session.context_ledger

    ledger.refresh(session.history_snapshot())
    ledger.refresh(session.history_snapshot())

    assert _head(session).generation == 2
    assert _head(session).node_id is None
    assert _admit(session, _operation(session, "a", "Hangzhou", generation=2)).status == "applied"


def test_unrelated_global_revision_does_not_reject_current_sources(session):
    session.register_history_item("a", _user("Shanghai"))
    candidate = _operation(session, "a", "Shanghai")
    session.register_history_item("b", _user("Unrelated text"))

    assert _admit(session, candidate).status == "applied"


def test_parent_invalidation_clears_card_head(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("b", _user("Tea"))
    _admit(session, _operation(session, "b", "Tea", key="user/self/drink", parents=(1,)))
    session.delete_history_item("a")

    session.context_ledger.refresh(session.history_snapshot())

    assert session.context_ledger.card_candidates() == ()
    assert _head(session, "user/self/drink").node_id is None


def test_duplicate_key_rejects_whole_batch(session):
    session.register_history_item("a", _user("Shanghai"))
    candidate = _operation(session, "a", "Shanghai")

    result = _admit(session, candidate, replace(candidate, text="Hangzhou"))

    assert result.status == "invalid"
    assert session.context_ledger.nodes() == ()
    assert _admit(session, candidate).node_ids == (1,)


def test_failed_replacement_does_not_retire_existing_head(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("b", _user("Hangzhou"))
    replacement = _operation(session, "b", "Hangzhou", generation=1)
    bad = replace(replacement, fact_key="user/self/food", expected_generation=0, parents=(999,))

    result = _admit(session, replacement, bad)

    assert result.status == "stale"
    assert _head(session).node_id == 1
    assert session.context_ledger.valid(1)
    assert _admit(session, replacement).node_ids == (2,)


def test_retract_keeps_tombstone_and_rejects_late_write(session):
    session.register_history_item("a", _user("Shanghai"))
    old_candidate = _operation(session, "a", "Shanghai")
    _admit(session, old_candidate)
    session.register_history_item("b", _user("Forget my current city."))

    result = _admit(session, _operation(session, "b", None, generation=1))

    assert result.status == "applied"
    assert result.node_ids == ()
    assert _head(session).generation == 2
    assert _head(session).node_id is None
    assert _admit(session, old_candidate).status == "stale"


def test_empty_batch_is_no_change(session):
    result = _admit(session)

    assert result.status == "no_change"
    assert result.node_ids == ()
    assert session.context_ledger.card_catalog() == ()


def test_oversized_batch_does_not_consume_ids(session):
    session.register_history_item("a", _user("Some committed text"))
    operations = [_operation(session, "a", "value", key=f"user/self/fact{index}") for index in range(9)]

    result = _admit(session, *operations)

    assert result.status == "capacity"
    assert session.context_ledger.nodes() == ()
    assert session.context_ledger.card_catalog() == ()
    assert _admit(session, operations[0]).node_ids == (1,)


def test_old_catalog_revision_cannot_create_new_key(session):
    session.register_history_item("a", _user("Shanghai and tea"))
    _admit(session, _operation(session, "a", "Shanghai"))

    result = _admit(session, _operation(session, "a", "Tea", key="user/self/drink"), catalog_revision=0)

    assert result.status == "stale"
    assert len(session.context_ledger.card_catalog()) == 1


def test_generic_node_admission_cannot_bypass_card_invariants(session):
    session.register_history_item("a", _user("Shanghai"))
    snapshot = session.history_snapshot()

    with pytest.raises(DuplexContextNodeRejectedError, match="memory_card_requires_batch_admission"):
        session.context_ledger.add_node(
            snapshot,
            level=DuplexContextLevel.MEMORY_CARD,
            sources=source_refs(snapshot, snapshot.source_ids[0]),
            message={"role": "user", "content": "Unchecked card"},
        )

    assert session.context_ledger.nodes() == ()
    assert _admit(session, _operation(session, "a", "Shanghai")).node_ids == (1,)


def test_batch_cannot_retire_a_parent_used_by_another_new_card(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("b", _user("Hangzhou and tea"))
    replacement = _operation(session, "b", "Hangzhou", generation=1)
    dependent = _operation(session, "b", "Tea", key="user/self/drink", parents=(1,))

    result = _admit(session, replacement, dependent)

    assert result.status == "invalid"
    assert _head(session).node_id == 1
    assert session.context_ledger.valid(1)
    assert _admit(session, replacement).node_ids == (2,)


def test_same_slot_edit_can_supply_fresh_evidence_after_invalidation(session):
    session.register_history_item("a", _user("Shanghai"))
    _admit(session, _operation(session, "a", "Shanghai"))
    session.register_history_item("a", _user("Hangzhou"))
    session.context_ledger.refresh(session.history_snapshot())

    result = _admit(session, _operation(session, "a", "Hangzhou", generation=2))

    assert result.status == "applied"
    assert _head(session).generation == 3
    assert [node.message["content"] for node in session.context_ledger.card_candidates()] == ["Hangzhou"]


def test_fact_slot_limit_includes_tombstones(session):
    session.register_history_item("a", _user("A collection of facts"))
    for start in range(0, 256, 8):
        result = _admit(
            session,
            *[_operation(session, "a", "value", key=f"user/self/fact{index}") for index in range(start, start + 8)],
        )
        assert result.status == "applied"
    session.delete_history_item("a")
    session.context_ledger.refresh(session.history_snapshot())
    session.register_history_item("b", _user("New evidence"))

    result = _admit(session, _operation(session, "b", "new", key="user/self/another"))

    assert result.status == "capacity"
    assert len(session.context_ledger.card_catalog()) == 256
    assert session.context_ledger.card_candidates() == ()
    assert _admit(session, _operation(session, "b", "updated", key="user/self/fact0", generation=2)).status == "applied"


@pytest.mark.parametrize(
    "changes",
    [
        {"fact_key": ""},
        {"fact_key": "x" * 129},
        {"fact_key": "城市" * 22},
        {"expected_generation": True},
        {"expected_generation": -1},
        {"text": ""},
        {"text": "x" * 513},
    ],
)
def test_invalid_card_fields_leave_no_writes(session, changes):
    session.register_history_item("a", _user("Shanghai"))

    result = _admit(session, replace(_operation(session, "a", "Shanghai"), **changes))

    assert result.status == "invalid"
    assert session.context_ledger.nodes() == ()
    assert session.context_ledger.card_catalog() == ()
