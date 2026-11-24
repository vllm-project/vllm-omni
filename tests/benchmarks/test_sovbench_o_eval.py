# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SOVBench-O scorer: option-letter extraction and accuracy parity (issue #10954).

Reproduces the reference scorer's behaviour on fixed predictions, including an
unparsable answer that must be excluded from the accuracy denominator.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.benchmarks.data_modules.sovbench_o_dataset import (
    SOVBenchOSampleRequest,
    _parse_session,
)
from vllm_omni.benchmarks.data_modules.sovbench_o_eval import (
    SOVBENCH_O_FAIL_MSG,
    compute_sovbench_o_accuracy_metrics,
    extract_sovbench_choice_letter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("C", "C"),
        ("  The answer is B.  ", "B"),
        ("(D)", "D"),
        ("A. because of the opening scene", "A"),
        ('{"answer": "E", "reason": "x"}', "E"),
        ("The final answer is \\boxed{A}", "A"),
        ("**B**", "B"),
        ("<answer>C</answer>", "C"),
        ("<answer> B </answer>", "B"),
        ("The correct option is C", "C"),
        ("Best answer: E", "E"),
        ("I think the answer is (A)", "A"),
        (SOVBENCH_O_FAIL_MSG, ""),
        ("", ""),
        (None, ""),
        # First-occurrence fallback when nothing else parses.
        ("option c seems right", "C"),
        ("好 B 好", "B"),
    ],
)
def test_extract_choice_letter(prediction: Any, expected: str) -> None:
    assert extract_sovbench_choice_letter(prediction) == expected


def test_extract_choice_letter_prefers_last_answer_tag() -> None:
    assert extract_sovbench_choice_letter("<answer>A</answer> then <answer>D</answer>") == "D"


def _turn(session_turn_id: int, group_id: int, turn_type: str, answer: str, *, group_num_turns: int) -> SimpleNamespace:
    return SimpleNamespace(
        session_turn_id=session_turn_id,
        session_num_turns=2,
        group_id=group_id,
        group_start_turn_id=0,
        step_id=session_turn_id,
        group_num_turns=group_num_turns,
        turn_type=turn_type,
        answer=answer,
    )


def _session(session_id: int, turns: list[SimpleNamespace], predictions: list[str]) -> SimpleNamespace:
    return SimpleNamespace(session_id=session_id, turns=turns, turn_predictions=list(predictions), num_turns=len(turns))


def _request(session: SimpleNamespace) -> SOVBenchOSampleRequest:
    return SOVBenchOSampleRequest(
        prompt="SOVBench-O session",
        prompt_len=0,
        expected_output_len=64,
        sovbench_session=session,
    )


def test_accuracy_metrics_match_reference_semantics() -> None:
    """Fixed predictions with one unparsable answer, scored by hand.

    Session 0 (one group of 2): Real-time C pred "C" -> 1; Recall A pred FAIL -> -1.
    Session 1 (two groups of 1): Real-time B pred "The answer is B" -> 1;
    Proactive D pred "(A)" -> 0.

    overall = (1 + 1 + 0) / 3, valid = 3, total = 4.
    by_turn_type: Real-time 1.0, Proactive 0.0, Recall {} (only invalid rows).
    by_group_size: 2 -> 1.0 (valid row), 1 -> 0.5.
    by_session_length: 2 -> overall.
    """
    session0 = _session(
        0,
        [_turn(0, 0, "Real-time", "C", group_num_turns=2), _turn(1, 0, "Recall", "A", group_num_turns=2)],
        ["C", SOVBENCH_O_FAIL_MSG],
    )
    session1 = _session(
        1,
        [_turn(0, 0, "Real-time", "B", group_num_turns=1), _turn(1, 1, "Proactive", "D", group_num_turns=1)],
        ["The answer is B", "(A)"],
    )
    requests = [_request(session0), _request(session1)]
    outputs = [SimpleNamespace(success=True), SimpleNamespace(success=True)]

    metrics = compute_sovbench_o_accuracy_metrics(requests, outputs, include_per_item=True)
    assert metrics is not None
    assert metrics["sovbench_o_total"] == 4
    assert metrics["sovbench_o_valid"] == 3
    assert metrics["sovbench_o_correct"] == 2
    assert metrics["sovbench_o_accuracy"] == pytest.approx(2 / 3)
    # Per-category breakdowns carry correct/valid/total counts, and a category
    # with no parseable predictions stays visible with a None accuracy.
    assert metrics["sovbench_o_by_turn_type"] == {
        "Real-time": {"accuracy": 1.0, "correct": 2, "valid": 2, "total": 2},
        "Recall": {"accuracy": None, "correct": 0, "valid": 0, "total": 1},
        "Proactive": {"accuracy": 0.0, "correct": 0, "valid": 1, "total": 1},
    }
    assert metrics["sovbench_o_by_group_size"] == {
        "1": {"accuracy": 0.5, "correct": 1, "valid": 2, "total": 2},
        "2": {"accuracy": 1.0, "correct": 1, "valid": 1, "total": 2},
    }
    assert metrics["sovbench_o_by_session_length"] == {
        "2": {"accuracy": pytest.approx(2 / 3), "correct": 2, "valid": 3, "total": 4},
    }

    items = metrics["sovbench_o_eval_items"]
    assert [item["score"] for item in items] == [1, -1, 1, 0]
    assert [item["pred_letter"] for item in items] == ["C", "", "B", "A"]


def test_accuracy_metrics_none_for_other_workloads() -> None:
    assert compute_sovbench_o_accuracy_metrics([], []) is None
    other = SimpleNamespace(prompt="x", prompt_len=1, expected_output_len=1)
    assert compute_sovbench_o_accuracy_metrics([other], [SimpleNamespace()]) is None


def test_all_unparsable_yields_nan_overall() -> None:
    session = _session(0, [_turn(0, 0, "Real-time", "C", group_num_turns=1)], [SOVBENCH_O_FAIL_MSG])
    metrics = compute_sovbench_o_accuracy_metrics([_request(session)], [SimpleNamespace()])
    assert metrics is not None
    assert metrics["sovbench_o_valid"] == 0
    assert metrics["sovbench_o_accuracy"] != metrics["sovbench_o_accuracy"]  # NaN


def test_answer_verification_against_reference_annotations() -> None:
    """The parser cross-checks assistant letters against correct_indices, like the reference."""
    item = {
        "messages": [
            {"role": "user", "content": "<video>Q?\n\nChoices:\n(A). x\n(B). y"},
            {"role": "assistant", "content": "B", "category": "Real-time"},
        ],
        "videos": ["benchvideos/s/s_0.00_1.00.mp4"],
        "entry_metadata": [{"correct_indices": [1]}],
    }
    parsed = _parse_session(item, session_id=0)
    assert parsed is not None
    assert parsed.turns[0].answer == "B"


def test_extract_letter_explicit_mode_rejects_any_letter_fallback() -> None:
    """The explicit mode keeps every structured form but drops the any-letter layer.

    The replay-history filter for length-limited turns uses it so a truncated
    question echo (whose only "letter" comes from words such as USER) cannot
    masquerade as an answer; the default scorer path is unchanged.
    """
    # Loose fallback would read a letter out of the echoed question.
    assert extract_sovbench_choice_letter("User question: what happens after the timer") != ""
    assert (
        extract_sovbench_choice_letter("User question: what happens after the timer", allow_any_letter_fallback=False)
        == ""
    )
    # Every explicit form still resolves in explicit mode.
    assert extract_sovbench_choice_letter("The answer is B", allow_any_letter_fallback=False) == "B"
    assert extract_sovbench_choice_letter('"answer": "C"', allow_any_letter_fallback=False) == "C"
    assert extract_sovbench_choice_letter("\boxed{A}", allow_any_letter_fallback=False) == "A"
    assert extract_sovbench_choice_letter("<answer>E</answer>", allow_any_letter_fallback=False) == "E"
    # The default (scorer) behavior is unchanged.
    assert extract_sovbench_choice_letter("The answer is B") == "B"
