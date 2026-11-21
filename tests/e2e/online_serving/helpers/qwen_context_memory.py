# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU-only oracles for observed Qwen context-memory model runs.

Records must come from the actual request/result and foreground submission
boundaries. Session IDs are protocol IDs; object addresses and time windows
cannot establish attribution. These checks do not manufacture missing evidence.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass


class FixtureInconclusiveError(RuntimeError):
    """The run did not exercise the intended fixture; never count it as passed.

    Keep this separate from model-answer and evidence-integrity assertion
    failures, and preserve the reason in the evaluation's case report.
    """


@dataclass(frozen=True)
class SourceRef:
    source_id: int
    revision: int


@dataclass(frozen=True)
class ResponseRecord:
    session_id: str
    request_id: str
    answer: str
    status: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SummaryRecord:
    """Join one auxiliary request to its accepted result by request ID.

    ``previous_node_id`` and ``previous_text`` come from the request, while
    ``node_id``, ``text`` and ``sources`` describe the published result.
    ``sources`` includes the summary's inherited source dependency closure.
    """

    session_id: str
    request_id: str
    node_id: int
    previous_node_id: int | None
    text: str
    accepted: bool
    sources: tuple[SourceRef, ...] = ()
    previous_text: str | None = None


@dataclass(frozen=True)
class ForegroundRecord:
    """The final submitted prompt, not the authoritative history snapshot.

    ``selected_sources`` contains only raw sources included in the prompt,
    never the summary's dependency closure. ``summary_node_ids`` has one
    entry for each actual summary insertion; do not deduplicate this tuple.
    The observer must verify these insertions against the rendered prompt.
    """

    session_id: str
    request_id: str
    prompt_token_ids: tuple[int, ...]
    selected_sources: tuple[SourceRef, ...]
    summary_node_ids: tuple[int, ...]


def normalize_answer(value: str) -> str:
    """Allow surrounding whitespace, terminal punctuation and case only."""
    return value.strip().rstrip(".。!").strip().casefold()


def _assert_attribution(actual_session: str, actual_request: str, session_id: str, request_id: str) -> None:
    assert session_id and request_id, "expected session/request attribution must be nonempty"
    assert actual_session == session_id, f"session attribution mismatch: {actual_session!r} != {session_id!r}"
    assert actual_request == request_id, f"request attribution mismatch: {actual_request!r} != {request_id!r}"


def assert_answer(
    response: ResponseRecord,
    *,
    session_id: str,
    request_id: str,
    expected_answers: Sequence[str],
) -> None:
    """Require a successful response and one predeclared complete answer."""
    _assert_attribution(response.session_id, response.request_id, session_id, request_id)
    assert response.status == "completed", f"response status is {response.status!r}"
    assert not response.errors, f"response has errors: {response.errors!r}"
    expected = {normalize_answer(answer) for answer in expected_answers}
    if not expected or "" in expected:
        raise ValueError("expected_answers must contain nonempty complete answers")
    actual = normalize_answer(response.answer)
    assert actual in expected, f"answer {actual!r} is not one of {sorted(expected)!r}"


def assert_summary_chain(
    records: Sequence[SummaryRecord], *, session_id: str, minimum_updates: int = 2
) -> tuple[SummaryRecord, ...]:
    """Validate an ordered sequence of distinct accepted summary publications.

    Supply accepted publications for the scenario, retaining duplicates so
    repeated log entries cannot masquerade as updates. Rejected attempts may
    be retained separately as diagnostic evidence. The first publication may
    itself replace an older node outside this scenario's observation window.
    """
    if minimum_updates < 2:
        raise ValueError("at least two summary updates are required")
    assert session_id, "expected session attribution must be nonempty"
    nodes: set[int] = set()
    requests: set[str] = set()
    previous: SummaryRecord | None = None
    for record in records:
        assert record.session_id == session_id, "summary session attribution mismatch"
        assert record.request_id, "summary request attribution is missing"
        assert record.accepted, "summary publication was not accepted"
        assert record.text.strip(), "accepted summary text is empty"
        assert record.sources, "accepted summary has no source dependencies"
        assert len(set(record.sources)) == len(record.sources), "duplicate summary source refs"
        assert len({ref.source_id for ref in record.sources}) == len(record.sources), "conflicting source revisions"
        assert record.node_id not in nodes, "duplicate summary node publication"
        assert record.request_id not in requests, "duplicate summary request publication"
        if previous is not None:
            assert record.node_id > previous.node_id, "summary node order is not increasing"
            assert record.previous_node_id == previous.node_id, "summary predecessor node mismatch"
            assert record.previous_text == previous.text, "summary predecessor text mismatch"
            assert set(previous.sources).issubset(record.sources), "summary lost or changed inherited source refs"
        nodes.add(record.node_id)
        requests.add(record.request_id)
        previous = record
    if len(records) < minimum_updates:
        raise FixtureInconclusiveError(f"only {len(records)} distinct accepted summary updates; need {minimum_updates}")
    return tuple(records)


def _contains_fact(text: str, fact: str) -> bool:
    """Match a complete fact without accepting a numeric prefix/superstring."""
    text, fact = text.casefold(), fact.casefold()
    if not fact:
        raise ValueError("fact must be nonempty")
    start = text.find(fact)
    while start >= 0:
        end = start + len(fact)
        left_boundary = start == 0 or not (text[start - 1].isalnum() and fact[0].isalnum())
        right_boundary = end == len(text) or not (text[end].isalnum() and fact[-1].isalnum())
        if left_boundary and right_boundary:
            return True
        start = text.find(fact, start + 1)
    return False


def assert_summary_facts(
    summary: SummaryRecord, *, required_facts: Sequence[str] = (), forbidden_facts: Sequence[str] = ()
) -> None:
    """Check fixture facts in generated text; this is not a semantic judge.

    In correction scenarios the old value must be entirely absent, including
    explanations that label it obsolete. Required values use word boundaries.
    """
    assert summary.accepted, "cannot check facts in a rejected summary"
    for fact in required_facts:
        assert _contains_fact(summary.text, fact), f"required fact {fact!r} missing from summary {summary.node_id}"
    for fact in forbidden_facts:
        assert not _contains_fact(summary.text, fact), f"forbidden fact {fact!r} in summary {summary.node_id}"


def assert_foreground(
    record: ForegroundRecord,
    *,
    session_id: str,
    request_id: str,
    excluded_source_ids: Collection[int],
    expected_summary: SummaryRecord | None,
) -> None:
    """Prove the submitted prompt depends on summary memory, not target raw turns.

    The no-summary budget control uses ``expected_summary=None``. A fixture
    retaining the target raw source cannot distinguish raw recall from memory
    recall, regardless of the model's answer. Exclude all copies of the fact,
    including assistant echoes, when selecting ``excluded_source_ids``.
    """
    _assert_attribution(record.session_id, record.request_id, session_id, request_id)
    if not record.prompt_token_ids:
        raise FixtureInconclusiveError("final foreground prompt token IDs were not captured")
    retained = {source.source_id for source in record.selected_sources}.intersection(excluded_source_ids)
    if retained:
        raise FixtureInconclusiveError(f"target raw sources remain in the foreground prompt: {sorted(retained)}")
    if expected_summary is None:
        assert not record.summary_node_ids, "budget control unexpectedly includes a summary"
    else:
        assert expected_summary.session_id == session_id, "foreground summary session attribution mismatch"
        assert expected_summary.accepted, "foreground references a rejected summary"
        assert record.summary_node_ids == (expected_summary.node_id,), "foreground must include expected summary once"
