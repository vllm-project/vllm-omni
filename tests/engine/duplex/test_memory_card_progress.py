# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Coverage is source/span/catalog-page work, not a whole-history revision."""

import pytest

from tests.engine.duplex.test_memory_card_ledger import _admit, _operation
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.memory_types import (
    DuplexMemoryEvidenceSelection,
    DuplexMemoryProposal,
    DuplexMemoryProposalBatch,
)
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def session():
    return DuplexEngineSession(session_id="progress", config=DuplexSessionConfig(model="test-model"))


def seed_catalog(session, size):
    session.register_history_item("a", {"role": "user", "content": "ABCDEFZ"})
    for start in range(0, size, 8):
        result = _admit(
            session,
            *(
                _operation(session, "a", "A fact.", key=f"user/self/fact{index:03}")
                for index in range(start, min(start + 8, size))
            ),
        )
        assert result.status == "applied"


def complete(session, context):
    result = session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch(()))
    assert result.status == "no_change"


def upsert(session, context, key):
    result = session.admit_memory_result(
        context=context,
        proposal=DuplexMemoryProposalBatch(
            (
                DuplexMemoryProposal(
                    key, "Updated fact.", (DuplexMemoryEvidenceSelection(0, 0, len(context.evidence[0].text)),)
                ),
            )
        ),
    )
    assert result.status == "applied"


def test_long_source_watermark_requires_final_span(session):
    session.register_history_item("a", {"role": "user", "content": "ABCDEFZ"})
    seen = []
    for expected in ((0, 3), (3, 6), (6, 7)):
        context = session.capture_memory_context(max_chars=3)
        assert context is not None
        assert (context.evidence[0].char_start, context.evidence[0].char_end) == expected
        seen.append(context.evidence[0].text)
        complete(session, context)
    assert "".join(seen) == "ABCDEFZ"
    assert session.capture_memory_context(max_chars=3) is None


def test_catalog_page_is_not_full_reconciliation(session):
    seed_catalog(session, 17)
    first = session.capture_memory_context()
    assert len(first.catalog) == 16
    complete(session, first)

    second = session.capture_memory_context()
    assert second.evidence == first.evidence
    assert len(second.catalog) == 1
    assert second.work.pass_id == first.work.pass_id
    assert second.work.catalog_page == first.work.catalog_page + 1
    assert {head.fact_key for head in first.catalog}.isdisjoint(head.fact_key for head in second.catalog)
    complete(session, second)
    assert session.capture_memory_context() is None


def test_own_admission_does_not_restart_catalog_pass(session):
    seed_catalog(session, 17)
    first = session.capture_memory_context()
    upsert(session, first, first.catalog[0].fact_key)

    second = session.capture_memory_context()
    assert second.work.pass_id == first.work.pass_id
    assert second.work.catalog_page == 1
    assert second.work.catalog_revision > first.work.catalog_revision
    assert len(second.catalog) == 1
    complete(session, second)
    assert session.capture_memory_context() is None


def test_new_key_adds_bounded_page_coverage(session):
    seed_catalog(session, 16)
    first = session.capture_memory_context()
    upsert(session, first, "user/self/newfact")

    extension = session.capture_memory_context()
    assert extension.work.pass_id == first.work.pass_id
    assert extension.work.catalog_page == 1
    assert [head.fact_key for head in extension.catalog] == ["user/self/newfact"]
    assert extension.evidence == first.evidence
    complete(session, extension)
    assert session.capture_memory_context() is None


def test_external_catalog_change_invalidates_pass(session):
    seed_catalog(session, 17)
    first = session.capture_memory_context()
    complete(session, first)
    second = session.capture_memory_context()
    head = session.context_ledger.card_catalog()[0]
    assert (
        _admit(
            session, _operation(session, "a", "Externally updated.", key=head.fact_key, generation=head.generation)
        ).status
        == "applied"
    )

    assert not session.memory_context_matches(second)
    restarted = session.capture_memory_context()
    assert restarted.work.pass_id != first.work.pass_id
    assert restarted.work.catalog_page == 0
    assert restarted.evidence == first.evidence


def test_edit_behind_catchup_cursor_prevents_completion(session):
    seed_catalog(session, 17)
    session.register_history_item("b", {"role": "user", "content": "Next evidence."})
    for _ in range(2):
        complete(session, session.capture_memory_context(max_sources=1))
    tail = session.capture_memory_context(max_sources=1)
    assert tail.evidence[0].text == "Next evidence."
    session.register_history_item("a", {"role": "user", "content": "Actually corrected earlier text."})

    restarted = session.capture_memory_context(max_sources=1)
    assert restarted.evidence[0].text == "Actually corrected earlier text."
    assert restarted.work.pass_id != tail.work.pass_id


def test_tail_append_does_not_reset_target_pass(session):
    seed_catalog(session, 17)
    first = session.capture_memory_context()
    session.register_history_item("b", {"role": "user", "content": "A later observation."})
    assert session.capture_memory_context() is first
    complete(session, first)
    second = session.capture_memory_context()
    assert second.work.pass_id == first.work.pass_id
    assert second.evidence == first.evidence
    complete(session, second)
    tail = session.capture_memory_context()
    assert tail.evidence[0].text == "A later observation."
