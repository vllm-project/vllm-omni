# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Semantic declarations require owner-issued authority and atomic admission."""

from dataclasses import replace

import pytest

from tests.engine.duplex.test_memory_card_progress import seed_catalog
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSpan, DuplexMemoryProposalBatch
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def session():
    return DuplexEngineSession(session_id="resolutions", config=DuplexSessionConfig(model="test-model"))


def block(session, *, keys=None):
    snapshot = session.history_snapshot()
    return session.memory_suppression.block(
        snapshot=snapshot,
        evidence=(DuplexMemoryEvidenceSpan(snapshot.source_ids[0], snapshot.source_revisions[0], 0, 7),),
        keys=keys,
        catalog=session.context_ledger.card_catalog(),
    )


def resolved(context, *, handle=0, keys=None, evidence=None, disposition="no_conflict"):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryResolution

    grant = context.barriers[0]
    return DuplexMemoryProposalBatch(
        (),
        (
            DuplexMemoryResolution(
                handle,
                grant.keys if keys is None else keys,
                grant.evidence if evidence is None else evidence,
                disposition,
            ),
        ),
    )


def test_plain_empty_result_never_resolves_correction(session):
    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    assert len(context.barriers) == 1
    assert session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch(())).status == "no_change"
    assert session.memory_suppression.suppress_all


def test_explicit_resolution_is_bound_to_captured_trigger(session):
    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    assert session.admit_memory_result(context=context, proposal=resolved(context)).status == "no_change"
    assert not session.memory_suppression.suppress_all


def test_catalog_page_resolution_cannot_release_global_barrier(session):
    seed_catalog(session, 17)
    block(session)
    first = session.capture_memory_context()
    assert len(first.barriers[0].keys) == 16
    assert session.admit_memory_result(context=first, proposal=resolved(first)).status == "no_change"
    assert session.memory_suppression.suppress_all
    second = session.capture_memory_context()
    assert len(second.barriers[0].keys) == 1
    assert session.admit_memory_result(context=second, proposal=resolved(second)).status == "no_change"
    assert not session.memory_suppression.suppress_all


def test_long_trigger_requires_resolution_of_every_text_fragment(session):
    seed_catalog(session, 1)
    block(session)
    for index in range(3):
        context = session.capture_memory_context(max_chars=3)
        assert session.admit_memory_result(context=context, proposal=resolved(context)).status == "no_change"
        assert session.memory_suppression.suppress_all == (index != 2)


@pytest.mark.parametrize("handle", [-1, 999, False])
def test_invalid_barrier_handle_does_not_consume_work_or_clear_barrier(session, handle):
    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    version = session.memory_suppression.version
    result = session.admit_memory_result(context=context, proposal=resolved(context, handle=handle))
    assert result.status == "invalid"
    assert session.memory_suppression.version == version
    assert session.capture_memory_context() is context


def test_resolution_cannot_claim_keys_outside_current_page(session):
    seed_catalog(session, 17)
    block(session)
    context = session.capture_memory_context()
    off_page = session.context_ledger.card_catalog()[-1].fact_key
    assert (
        session.admit_memory_result(context=context, proposal=resolved(context, keys=frozenset({off_page}))).status
        == "invalid"
    )
    assert session.memory_suppression.suppress_all


def test_invalid_late_resolution_cannot_partially_resolve_or_admit_card(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSelection, DuplexMemoryProposal

    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    valid = resolved(context).resolutions[0]
    proposal = DuplexMemoryProposalBatch(
        (DuplexMemoryProposal("user/self/new", "New fact.", (DuplexMemoryEvidenceSelection(0, 0, 7),)),),
        (valid, replace(valid, handle=999)),
    )
    version = session.memory_suppression.version
    before = session.context_ledger.nodes()
    assert session.admit_memory_result(context=context, proposal=proposal).status == "invalid"
    assert session.memory_suppression.version == version
    assert session.context_ledger.nodes() == before
    assert session.capture_memory_context() is context


def test_invalid_card_cannot_commit_valid_resolution(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSelection, DuplexMemoryProposal

    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    proposal = replace(
        resolved(context),
        operations=(DuplexMemoryProposal("user/self/new", "X" * 513, (DuplexMemoryEvidenceSelection(0, 0, 7),)),),
    )
    version = session.memory_suppression.version
    assert session.admit_memory_result(context=context, proposal=proposal).status == "invalid"
    assert session.memory_suppression.version == version
    assert session.memory_suppression.suppress_all


def test_new_barrier_fences_inflight_resolution(session):
    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    block(session, keys=frozenset({context.catalog[0].fact_key}))
    assert session.admit_memory_result(context=context, proposal=resolved(context)).status == "stale"
    assert session.memory_suppression.suppress_all


@pytest.mark.parametrize("disposition", [[], False, None, "unknown"])
def test_resolution_disposition_is_strictly_typed(session, disposition):
    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    assert (
        session.admit_memory_result(context=context, proposal=resolved(context, disposition=disposition)).status
        == "invalid"
    )
    assert session.memory_suppression.suppress_all


@pytest.mark.parametrize("disposition", ["replace", "retract"])
def test_resolution_mutation_requires_matching_operation(session, disposition):
    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    assert (
        session.admit_memory_result(context=context, proposal=resolved(context, disposition=disposition)).status
        == "invalid"
    )
    assert session.memory_suppression.suppress_all


def test_card_replacement_and_resolution_commit_together(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSelection, DuplexMemoryProposal

    seed_catalog(session, 1)
    block(session)
    context = session.capture_memory_context()
    key = context.catalog[0].fact_key
    proposal = replace(
        resolved(context, disposition="replace"),
        operations=(DuplexMemoryProposal(key, "Updated fact.", (DuplexMemoryEvidenceSelection(0, 0, 7),)),),
    )
    assert session.admit_memory_result(context=context, proposal=proposal).status == "applied"
    assert not session.memory_suppression.suppress_all
    assert session.context_ledger.card_catalog()[0].generation == 2
    assert session.context_ledger.card_candidates()[0].message["content"] == "Updated fact."


def test_new_key_requires_extension_proof_before_global_release(session):
    from vllm_omni.engine.duplex.memory_types import DuplexMemoryEvidenceSelection, DuplexMemoryProposal

    seed_catalog(session, 1)
    block(session)
    first = session.capture_memory_context()
    proposal = replace(
        resolved(first),
        operations=(DuplexMemoryProposal("user/self/new", "New fact.", (DuplexMemoryEvidenceSelection(0, 0, 7),)),),
    )
    assert session.admit_memory_result(context=first, proposal=proposal).status == "applied"
    assert session.memory_suppression.suppress_all
    extension = session.capture_memory_context()
    assert extension.barriers[0].keys == frozenset({"user/self/new"})
    assert session.admit_memory_result(context=extension, proposal=resolved(extension)).status == "no_change"
    assert not session.memory_suppression.suppress_all
