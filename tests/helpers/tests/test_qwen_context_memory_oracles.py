# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Negative controls for the real-model context-memory evaluation oracles."""

from dataclasses import replace

import pytest

from tests.e2e.online_serving.helpers.qwen_context_memory import (
    FixtureInconclusiveError,
    ForegroundRecord,
    ResponseRecord,
    SourceRef,
    SummaryRecord,
    assert_answer,
    assert_foreground,
    assert_summary_chain,
    assert_summary_facts,
    normalize_answer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SOURCE = SourceRef(3, 0)
_FIRST = SummaryRecord("session-a", "summary-1", 5, None, "The locker code is 8632.", True, (_SOURCE,))
_SECOND = SummaryRecord("session-a", "summary-2", 6, 5, "Locker code: 8632.", True, (_SOURCE,), _FIRST.text)
_RESPONSE = ResponseRecord("session-a", "foreground-1", "8632", "completed")
_FOREGROUND = ForegroundRecord("session-a", "foreground-1", (10, 20, 30), (SourceRef(9, 0),), (6,))


@pytest.mark.parametrize("answer", ["8632", " 8632. ", "8632。", "8632!"])
def test_numeric_answer_accepts_only_complete_normalized_value(answer):
    assert_answer(
        replace(_RESPONSE, answer=answer),
        session_id="session-a",
        request_id="foreground-1",
        expected_answers=("8632",),
    )


@pytest.mark.parametrize("answer", ["18632", "86320", "8632 is obsolete", "The code is 8632", "8632 or 4271", ""])
def test_numeric_answer_rejects_substrings_and_explanations(answer):
    with pytest.raises(AssertionError, match="answer"):
        assert_answer(
            replace(_RESPONSE, answer=answer),
            session_id="session-a",
            request_id="foreground-1",
            expected_answers=("8632",),
        )


@pytest.mark.parametrize("answer", ["not unknown", "UNKNOWN but it was 7193", "unknown code"])
def test_unknown_answer_rejects_negation_and_explanations(answer):
    with pytest.raises(AssertionError, match="answer"):
        assert_answer(
            replace(_RESPONSE, answer=answer),
            session_id="session-a",
            request_id="foreground-1",
            expected_answers=("UNKNOWN",),
        )


def test_unknown_answer_accepts_only_predeclared_complete_alternatives():
    assert normalize_answer(" UNKNOWN! ") == "unknown"
    assert_answer(
        replace(_RESPONSE, answer="Unavailable."),
        session_id="session-a",
        request_id="foreground-1",
        expected_answers=("UNKNOWN", "unavailable"),
    )


@pytest.mark.parametrize(
    "response",
    [
        replace(_RESPONSE, status="failed"),
        replace(_RESPONSE, status="cancelled"),
        replace(_RESPONSE, status="incomplete"),
        replace(_RESPONSE, errors=("context_changed",)),
        replace(_RESPONSE, session_id="session-b"),
        replace(_RESPONSE, request_id="other-request"),
    ],
)
def test_answer_alone_cannot_hide_failure_or_wrong_attribution(response):
    with pytest.raises(AssertionError):
        assert_answer(response, session_id="session-a", request_id="foreground-1", expected_answers=("8632",))


def test_two_distinct_accepted_updates_link_the_actual_previous_summary():
    assert assert_summary_chain((_FIRST, _SECOND), session_id="session-a") == (_FIRST, _SECOND)


@pytest.mark.parametrize(
    "second",
    [
        _FIRST,
        replace(_SECOND, node_id=5),
        replace(_SECOND, request_id="summary-1"),
        replace(_SECOND, session_id="session-b"),
        replace(_SECOND, previous_node_id=4),
        replace(_SECOND, previous_text="An unrelated summary."),
        replace(_SECOND, accepted=False),
        replace(_SECOND, text=""),
        replace(_SECOND, node_id=4),
        replace(_SECOND, sources=()),
        replace(_SECOND, sources=(SourceRef(8, 0),)),
        replace(_SECOND, sources=(SourceRef(3, 1),)),
        replace(_SECOND, sources=(_SOURCE, _SOURCE)),
        replace(_SECOND, sources=(_SOURCE, SourceRef(3, 1))),
    ],
)
def test_summary_event_count_cannot_hide_invalid_chain(second):
    with pytest.raises(AssertionError):
        assert_summary_chain((_FIRST, second), session_id="session-a")


@pytest.mark.parametrize("records", [(), (_FIRST,)])
def test_missing_summary_trigger_is_inconclusive_not_model_failure(records):
    with pytest.raises(FixtureInconclusiveError, match="updates"):
        assert_summary_chain(records, session_id="session-a")


@pytest.mark.parametrize("sources", [(), (_SOURCE, _SOURCE), (_SOURCE, SourceRef(3, 1))])
def test_first_summary_requires_unambiguous_source_revisions(sources):
    with pytest.raises(AssertionError, match="source"):
        assert_summary_chain((replace(_FIRST, sources=sources), _SECOND), session_id="session-a")


def test_summary_update_can_inherit_previous_sources_and_add_new_sources():
    second = replace(_SECOND, sources=(_SOURCE, SourceRef(8, 0)))
    assert assert_summary_chain((_FIRST, second), session_id="session-a") == (_FIRST, second)


def test_summary_fact_checks_reject_stale_facts_and_numeric_superstrings():
    assert_summary_facts(_SECOND, required_facts=("8632",), forbidden_facts=("4271",))
    with pytest.raises(AssertionError, match="forbidden"):
        assert_summary_facts(
            replace(_SECOND, text="The code is 8632; 4271 is obsolete."),
            required_facts=("8632",),
            forbidden_facts=("4271",),
        )
    with pytest.raises(AssertionError, match="required"):
        assert_summary_facts(replace(_SECOND, text="The code is 18632."), required_facts=("8632",))


def test_foreground_uses_summary_once_and_excludes_target_raw_source():
    assert_foreground(
        _FOREGROUND,
        session_id="session-a",
        request_id="foreground-1",
        excluded_source_ids=(3,),
        expected_summary=_SECOND,
    )


@pytest.mark.parametrize(
    "foreground",
    [
        replace(_FOREGROUND, session_id="session-b"),
        replace(_FOREGROUND, request_id="other-request"),
        replace(_FOREGROUND, summary_node_ids=()),
        replace(_FOREGROUND, summary_node_ids=(5,)),
        replace(_FOREGROUND, summary_node_ids=(6, 6)),
        replace(_FOREGROUND, summary_node_ids=(5, 6)),
    ],
)
def test_foreground_rejects_wrong_attribution_missing_stale_or_repeated_summary(foreground):
    with pytest.raises(AssertionError):
        assert_foreground(
            foreground,
            session_id="session-a",
            request_id="foreground-1",
            excluded_source_ids=(3,),
            expected_summary=_SECOND,
        )


@pytest.mark.parametrize("revision", [0, 1])
def test_target_raw_still_selected_invalidates_fixture_even_at_new_revision(revision):
    with pytest.raises(FixtureInconclusiveError, match="raw"):
        assert_foreground(
            replace(_FOREGROUND, selected_sources=(_SOURCE, SourceRef(3, revision))),
            session_id="session-a",
            request_id="foreground-1",
            excluded_source_ids=(3,),
            expected_summary=_SECOND,
        )


def test_foreground_requires_final_prompt_token_evidence():
    with pytest.raises(FixtureInconclusiveError, match="token"):
        assert_foreground(
            replace(_FOREGROUND, prompt_token_ids=()),
            session_id="session-a",
            request_id="foreground-1",
            excluded_source_ids=(3,),
            expected_summary=_SECOND,
        )


def test_budget_control_requires_no_summary_in_actual_foreground():
    assert_foreground(
        replace(_FOREGROUND, summary_node_ids=()),
        session_id="session-a",
        request_id="foreground-1",
        excluded_source_ids=(3,),
        expected_summary=None,
    )
    with pytest.raises(AssertionError, match="summary"):
        assert_foreground(
            _FOREGROUND,
            session_id="session-a",
            request_id="foreground-1",
            excluded_source_ids=(3,),
            expected_summary=None,
        )
