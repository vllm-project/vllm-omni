# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Recovery requires explicit coverage of a frozen, current source horizon."""

import pytest

from tests.engine.duplex.test_memory_card_ledger import _admit, _operation
from tests.engine.duplex.test_memory_card_progress import seed_catalog
from tests.engine.duplex.test_memory_card_resolution import block
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.memory_types import DuplexMemoryProposalBatch
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def session():
    return DuplexEngineSession(session_id="reconcile", config=DuplexSessionConfig(model="test-model"))


def edited_trigger(session, *, cards=1):
    seed_catalog(session, cards)
    block(session)
    session.register_history_item("a", {"role": "user", "content": "New text."})


def affirm(session, context):
    result = session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch((), reconciled=True))
    assert result.status == "no_change"


def overflow(session):
    seed_catalog(session, 1)
    for index in range(257):
        block(session, keys=frozenset({f"user/self/key{index}"}))
    assert session.memory_suppression.suppress_all


def test_edited_trigger_requires_explicit_current_text_reconciliation(session):
    edited_trigger(session)
    context = session.capture_memory_context()
    assert context.work.reconciliation_id is not None
    assert context.evidence[0].text == "New text."
    assert session.memory_suppression.suppress_all
    affirm(session, context)
    assert not session.memory_suppression.suppress_all
    assert session.capture_memory_context() is None


def test_empty_result_does_not_authorize_recovery_or_loop_forever(session):
    edited_trigger(session)
    context = session.capture_memory_context()
    assert session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch(())).status == "no_change"
    assert session.memory_suppression.suppress_all
    assert session.capture_memory_context() is None
    session.restart_memory_reconciliation()
    retry = session.capture_memory_context()
    assert retry.work.reconciliation_id != context.work.reconciliation_id
    affirm(session, retry)
    assert not session.memory_suppression.suppress_all


def test_deleted_only_trigger_needs_explicit_empty_authoritative_scan(session):
    seed_catalog(session, 1)
    block(session)
    session.delete_history_item("a")
    context = session.capture_memory_context()
    assert context is not None
    assert context.work.reconciliation_id is not None
    assert context.evidence == ()
    affirm(session, context)
    assert not session.memory_suppression.suppress_all
    assert session.context_ledger.card_candidates() == ()


def test_overflow_cannot_clear_before_final_source_fragment(session):
    overflow(session)
    seen = []
    for index in range(3):
        context = session.capture_memory_context(max_chars=3)
        seen.append(context.evidence[0].text)
        affirm(session, context)
        assert session.memory_suppression.suppress_all == (index != 2)
    assert "".join(seen) == "ABCDEFZ"


def test_recovery_requires_every_catalog_page(session):
    edited_trigger(session, cards=17)
    first = session.capture_memory_context()
    assert len(first.catalog) == 16
    affirm(session, first)
    assert session.memory_suppression.suppress_all
    second = session.capture_memory_context()
    assert len(second.catalog) == 1
    assert second.work.reconciliation_id == first.work.reconciliation_id
    affirm(session, second)
    assert not session.memory_suppression.suppress_all


def test_rewrite_behind_recovery_cursor_invalidates_completion(session):
    overflow(session)
    session.register_history_item("b", {"role": "user", "content": "Later text."})
    first = session.capture_memory_context(max_sources=1)
    affirm(session, first)
    tail = session.capture_memory_context(max_sources=1)
    assert tail.evidence[0].text == "Later text."
    session.register_history_item("a", {"role": "user", "content": "Edited behind cursor."})
    assert (
        session.admit_memory_result(context=tail, proposal=DuplexMemoryProposalBatch((), reconciled=True)).status
        == "stale"
    )
    restarted = session.capture_memory_context(max_sources=1)
    assert restarted.evidence[0].text == "Edited behind cursor."
    assert restarted.work.rewrite_epoch != tail.work.rewrite_epoch
    assert session.memory_suppression.suppress_all


def test_tail_append_does_not_move_recovery_target(session):
    overflow(session)
    first = session.capture_memory_context(max_chars=3)
    horizon = first.work.reconciliation_target
    session.register_history_item("b", {"role": "user", "content": "New tail."})
    assert session.capture_memory_context(max_chars=3) is first
    affirm(session, first)
    for _ in range(2):
        context = session.capture_memory_context(max_chars=3)
        assert context.work.reconciliation_id == first.work.reconciliation_id
        assert context.work.reconciliation_target == horizon
        affirm(session, context)
    assert not session.memory_suppression.suppress_all
    assert session.capture_memory_context(max_chars=3).evidence[0].text == "New"


def test_new_barrier_beyond_horizon_survives_completed_recovery(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSpan

    overflow(session)
    first = session.capture_memory_context()
    # Free one tracked barrier before creating a later barrier. Overflow state
    # itself stays set, and the recovery cutoff must not swallow this new ID.
    old = session.memory_suppression.barriers()[0]
    assert session.memory_suppression.resolve(
        snapshot=session.history_snapshot(),
        catalog=session.context_ledger.card_catalog(),
        barrier_id=old.barrier_id,
        generation=old.generation,
        evidence=old.evidence,
        keys=old.keys,
    )
    session.register_history_item("b", {"role": "user", "content": "Later correction."})
    snapshot = session.history_snapshot()
    later = session.memory_suppression.block(
        snapshot=snapshot,
        evidence=(DuplexMemoryEvidenceSpan(snapshot.source_ids[-1], snapshot.source_revisions[-1], 0, 17),),
        keys=None,
        catalog=session.context_ledger.card_catalog(),
    )
    assert later is not None
    current = session.capture_memory_context()
    assert current.work.reconciliation_target == first.work.reconciliation_target
    affirm(session, current)
    assert later in session.memory_suppression.barriers()
    assert session.memory_suppression.suppress_all


def test_external_catalog_change_restarts_full_recovery_coverage(session):
    overflow(session)
    first = session.capture_memory_context(max_chars=3)
    affirm(session, first)
    second = session.capture_memory_context(max_chars=3)
    head = session.context_ledger.card_catalog()[0]
    assert (
        _admit(
            session, _operation(session, "a", "Changed elsewhere.", key=head.fact_key, generation=head.generation)
        ).status
        == "applied"
    )
    assert (
        session.admit_memory_result(context=second, proposal=DuplexMemoryProposalBatch((), reconciled=True)).status
        == "stale"
    )
    restarted = session.capture_memory_context(max_chars=3)
    assert restarted.evidence[0].char_start == 0
    assert restarted.work.rewrite_epoch != second.work.rewrite_epoch


@pytest.mark.parametrize("value", [1, "true", None, []])
def test_reconciliation_assertion_requires_boolean(session, value):
    edited_trigger(session)
    context = session.capture_memory_context()
    result = session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch((), reconciled=value))
    assert result.status == "invalid"
    assert session.memory_suppression.suppress_all


def test_reconciliation_assertion_without_owner_grant_is_invalid(session):
    seed_catalog(session, 1)
    context = session.capture_memory_context()
    result = session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch((), reconciled=True))
    assert result.status == "invalid"


def test_one_unaffirmed_fragment_keeps_recovery_parked(session):
    overflow(session)
    first = session.capture_memory_context(max_chars=3)
    assert session.admit_memory_result(context=first, proposal=DuplexMemoryProposalBatch(())).status == "no_change"
    for _ in range(2):
        affirm(session, session.capture_memory_context(max_chars=3))
    assert session.memory_suppression.suppress_all
    assert session.capture_memory_context(max_chars=3) is None


def test_new_key_requires_replay_of_earlier_sources(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSelection, DuplexMemoryProposal

    overflow(session)
    session.register_history_item("b", {"role": "user", "content": "Later evidence."})
    first = session.capture_memory_context(max_sources=1)
    affirm(session, first)
    second = session.capture_memory_context(max_sources=1)
    proposal = DuplexMemoryProposalBatch(
        (DuplexMemoryProposal("user/self/new", "A new fact.", (DuplexMemoryEvidenceSelection(0, 0, 15),)),),
        reconciled=True,
    )
    assert session.admit_memory_result(context=second, proposal=proposal).status == "applied"
    extension = session.capture_memory_context(max_sources=1)
    assert extension.evidence[0].text == "Later evidence."
    assert [head.fact_key for head in extension.catalog] == ["user/self/new"]
    affirm(session, extension)
    assert session.memory_suppression.suppress_all
    replay = session.capture_memory_context(max_sources=1)
    assert replay.evidence[0].text == "ABCDEFZ"
    assert [head.fact_key for head in replay.catalog] == ["user/self/new"]
    assert replay.work.reconciliation_id == first.work.reconciliation_id
    affirm(session, replay)
    assert session.memory_suppression.suppress_all
    affirm(session, session.capture_memory_context(max_sources=1))
    assert not session.memory_suppression.suppress_all


def test_own_replacement_preserves_recovery_progress(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSelection, DuplexMemoryProposal

    overflow(session)
    first = session.capture_memory_context(max_chars=3)
    key = first.catalog[0].fact_key
    proposal = DuplexMemoryProposalBatch(
        (DuplexMemoryProposal(key, "Updated fact.", (DuplexMemoryEvidenceSelection(0, 0, 3),)),),
        reconciled=True,
    )
    assert session.admit_memory_result(context=first, proposal=proposal).status == "applied"
    second = session.capture_memory_context(max_chars=3)
    assert second.evidence[0].char_start == 3
    assert second.work.rewrite_epoch == first.work.rewrite_epoch
    assert second.work.reconciliation_id == first.work.reconciliation_id
    affirm(session, second)
    affirm(session, session.capture_memory_context(max_chars=3))
    assert not session.memory_suppression.suppress_all


def test_failed_recovery_batch_does_not_consume_confirmation(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSelection, DuplexMemoryProposal

    edited_trigger(session)
    context = session.capture_memory_context()
    proposal = DuplexMemoryProposalBatch(
        (DuplexMemoryProposal("user/self/new", "X" * 513, (DuplexMemoryEvidenceSelection(0, 0, 3),)),),
        reconciled=True,
    )
    assert session.admit_memory_result(context=context, proposal=proposal).status == "invalid"
    assert session.memory_suppression.suppress_all
    assert session.capture_memory_context() is context
    affirm(session, context)
    assert not session.memory_suppression.suppress_all


def test_internal_config_generation_does_not_restart_recovery(session):
    overflow(session)
    context = session.capture_memory_context(max_chars=3)
    session.config_generation += 1
    assert session.capture_memory_context(max_chars=3) is context
    affirm(session, context)
    next_context = session.capture_memory_context(max_chars=3)
    assert next_context.work.reconciliation_id == context.work.reconciliation_id
    assert next_context.evidence[0].char_start == 3


@pytest.mark.parametrize("recovering", [False, True])
def test_epoch_fence_preserves_completed_catalog_coverage(session, recovering):
    if recovering:
        edited_trigger(session, cards=17)
    else:
        seed_catalog(session, 17)
    first = session.capture_memory_context()
    result = session.admit_memory_result(context=first, proposal=DuplexMemoryProposalBatch((), reconciled=recovering))
    assert result.status == "no_change"
    pending = session.capture_memory_context()
    assert pending.work.catalog_page == 1
    session.epoch += 1
    assert not session.memory_context_matches(pending)
    retry = session.capture_memory_context()
    assert retry.work.epoch == session.epoch
    assert retry.work_key == pending.work_key
    assert retry.work.catalog_page == 1
    assert retry.work.rewrite_epoch == pending.work.rewrite_epoch
    assert (
        session.admit_memory_result(context=retry, proposal=DuplexMemoryProposalBatch((), reconciled=recovering)).status
        == "no_change"
    )
    assert session.capture_memory_context() is None
