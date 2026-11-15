# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Only the owner can replace oversized work; resizing proves no coverage."""

from dataclasses import replace

import pytest

from tests.engine.duplex.test_memory_card_progress import complete, seed_catalog
from tests.engine.duplex.test_memory_card_progress import session as _session_fixture
from vllm_omni.engine.duplex.memory_types import DuplexMemoryProposalBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
session = _session_fixture


def test_rechunk_reissues_work_without_covering_text_or_mutating_history(session):
    session.register_history_item("a", {"role": "user", "content": "abcdefgh"})
    before = session.history_snapshot()
    first = session.capture_memory_context(max_chars=8)
    assert session.rechunk_memory_context(first)
    assert not session.memory_context_matches(first)
    assert session.admit_memory_result(context=first, proposal=DuplexMemoryProposalBatch(())).status == "stale"
    second = session.capture_memory_context(max_chars=8)
    assert second.work.start == first.work.start
    assert second.work.end < first.work.end
    assert second.evidence[0].text == "abcd"
    assert second.work_key != first.work_key
    assert session.history_snapshot() == before
    assert not session.context_ledger.card_candidates()
    complete(session, second)
    third = session.capture_memory_context(max_chars=8)
    assert third.evidence[0].text == "efgh"
    complete(session, third)
    assert session.capture_memory_context(max_chars=8) is None


@pytest.mark.parametrize("change", ["copy", "epoch", "revision", "config", "closed"])
def test_rechunk_rejects_unretained_or_stale_capture(session, change):
    session.register_history_item("a", {"role": "user", "content": "abcdefgh"})
    context = session.capture_memory_context()
    if change == "copy":
        context = replace(context)
    elif change == "epoch":
        session.epoch += 1
    elif change == "revision":
        session.register_history_item("a", {"role": "user", "content": "edited"})
    elif change == "config":
        session.config_revision += 1
    else:
        from vllm_omni.engine.duplex.session.engine_session import DuplexSessionState

        session.state = DuplexSessionState.CLOSED
    assert not session.rechunk_memory_context(context)


def test_catalog_shrinks_before_text_and_all_pages_still_cover_same_source(session):
    seed_catalog(session, 17)
    context = session.capture_memory_context()
    original_text = context.evidence
    assert session.rechunk_memory_context(context)
    context = session.capture_memory_context()
    assert len(context.catalog) == 8 and context.evidence == original_text
    seen: list[str] = []
    while context is not None:
        seen.extend(head.fact_key for head in context.catalog)
        assert context.evidence == original_text
        complete(session, context)
        context = session.capture_memory_context()
    assert len(seen) == len(set(seen)) == 17


def test_rechunk_is_bounded_and_minimum_unfit_work_parks(session):
    seed_catalog(session, 17)
    for _ in range(20):
        context = session.capture_memory_context()
        if not session.rechunk_memory_context(context):
            break
    else:
        pytest.fail("unbounded resizing")
    assert len(context.catalog) == 1
    assert sum(len(item.text) for item in context.evidence) == 1
    assert session.capture_memory_context() is context
    assert not session.rechunk_memory_context(context)


def test_rechunk_does_not_rewind_completed_source_progress(session):
    session.register_history_item("a", {"role": "user", "content": "abcdefgh"})
    first = session.capture_memory_context(max_chars=4)
    complete(session, first)
    second = session.capture_memory_context(max_chars=4)
    assert session.rechunk_memory_context(second)
    smaller = session.capture_memory_context(max_chars=4)
    assert smaller.evidence[0].text == "ef"
    assert smaller.work.start == second.work.start


def test_rechunked_reconciliation_still_requires_every_text_fragment_and_catalog_page(session):
    from tests.engine.duplex.test_memory_card_reconciliation import affirm, edited_trigger

    edited_trigger(session, cards=17)
    context = session.capture_memory_context()
    target = context.work.reconciliation_target
    while len(context.catalog) > 1:
        assert session.rechunk_memory_context(context)
        assert session.memory_suppression.suppress_all
        context = session.capture_memory_context()
    assert session.rechunk_memory_context(context)
    assert session.memory_suppression.suppress_all
    fragments: dict[str, list[str]] = {}
    for _ in range(100):
        context = session.capture_memory_context()
        if context is None:
            break
        assert session.memory_suppression.suppress_all
        assert context.work.reconciliation_target == target
        for head in context.catalog:
            fragments.setdefault(head.fact_key, []).append(context.evidence[0].text)
        affirm(session, context)
    else:
        pytest.fail("reconciliation failed to finish")
    assert len(fragments) == 17
    assert all("".join(parts) == "New text." for parts in fragments.values())
    assert not session.memory_suppression.suppress_all
