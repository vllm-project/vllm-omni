# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scoring for the SOVBench-O streaming omni benchmark.

Ports the reference VLMEvalKit scorer from the StreamOV repository
(``streamov_bench/streaming_omni.py``, ``evaluate`` and
``_extract_choice_letter`` at commit 3decd08a): the option-letter extraction
cascade, exact-match scoring, unparsable predictions scored ``-1`` and
excluded from the accuracy denominator, and the per-category breakdowns
(turn type / group size / session length).
"""

from __future__ import annotations

from typing import Any

import regex as re

SOVBENCH_O_FAIL_MSG = "Failed to obtain answer via API."

_ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
_ANSWER_JSON_RE = re.compile(r'"answer"\s*:\s*"([ABCDE])"', re.IGNORECASE)
_BOXED_LETTER_PATTERNS = (
    re.compile(r"\\boxed\s*\{\s*([ABCDE])\s*\}", re.IGNORECASE),
    re.compile(r"\\box\s*\{\s*([ABCDE])\s*\}", re.IGNORECASE),
    re.compile(r"\*+\s*([ABCDE])\s*\*+", re.IGNORECASE),
)
# Removed from an upper-cased copy before the bare-letter patterns run, so a
# phrase such as "the answer is B" leaves only the letter to match.
_ANSWER_PHRASES = (
    "THE ANSWER IS",
    "ANSWER IS",
    "THE CORRECT ANSWER IS",
    "THE CORRECT OPTION IS",
    "THE BEST ANSWER IS",
    "BEST ANSWER",
    "BEST OPTION",
    "ANSWER",
    "OPTION",
)
_LETTER_PATTERNS = (
    re.compile(r"\(([ABCDE])\)"),
    re.compile(r"\b([ABCDE])\b"),
    re.compile(r"^\s*([ABCDE])\s*[.)]"),
)
_ANY_LETTER_RE = re.compile(r"[ABCDE]")


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def extract_sovbench_choice_letter(prediction: object) -> str:
    """Extract the option letter (A-E) from a model prediction.

    Faithful port of the reference ``_extract_choice_letter``: ``<answer>``
    tags (last match), JSON-style ``"answer": "A"``, ``\\boxed``/``\\box``/bold
    forms, phrase-stripped letter patterns, and finally the first A-E
    character anywhere in the output. Empty or failed predictions return "".
    """
    s = _safe_str(prediction).strip()
    if not s or SOVBENCH_O_FAIL_MSG in s:
        return ""

    answers = _ANSWER_TAG_RE.findall(s)
    if answers:
        s = _safe_str(answers[-1]).strip()

    m_json = _ANSWER_JSON_RE.search(s)
    if m_json:
        return m_json.group(1).upper()

    for pattern in _BOXED_LETTER_PATTERNS:
        m = pattern.search(s)
        if m:
            return m.group(1).upper()

    s_up = s.upper()
    for prefix in _ANSWER_PHRASES:
        s_up = s_up.replace(prefix, " ")

    for pattern in _LETTER_PATTERNS:
        m = pattern.search(s_up)
        if m:
            return m.group(1)

    m = _ANY_LETTER_RE.search(s_up)
    return m.group(0) if m else ""


def _mean(scores: list[int]) -> float:
    """Mean over non-negative scores; NaN when nothing is parseable."""
    valid = [s for s in scores if s >= 0]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _category_stats(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Per-value accuracy plus correct/valid/total counts for ``key``.

    Every value that occurs is kept even when none of its predictions parse,
    so a category never silently disappears from the breakdown; its accuracy is
    ``None`` in that case, matching the reference NaN overall.
    """
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = counts.setdefault(_safe_str(record.get(key)), {"correct": 0, "valid": 0, "total": 0})
        bucket["total"] += 1
        if record["score"] >= 0:
            bucket["valid"] += 1
            bucket["correct"] += int(record["score"])
    return {
        value: {
            "accuracy": (bucket["correct"] / bucket["valid"]) if bucket["valid"] else None,
            "correct": bucket["correct"],
            "valid": bucket["valid"],
            "total": bucket["total"],
        }
        for value, bucket in sorted(counts.items())
    }


def compute_sovbench_o_accuracy_metrics(
    input_requests: list[Any],
    outputs: list[Any],
    *,
    include_per_item: bool = False,
) -> dict[str, Any] | None:
    """If all requests are SOVBench-O session requests, compute the reference metrics.

    Returns ``None`` for any other workload, so the shared post-run hook stays a
    no-op outside SOVBench-O runs. Unparsable predictions score ``-1`` and are
    excluded from every accuracy denominator, matching the reference scorer.
    """
    # Imported here to keep the eval module importable without the dataset stack.
    from vllm_omni.benchmarks.data_modules.sovbench_o_dataset import SOVBenchOSampleRequest

    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(r, SOVBenchOSampleRequest) for r in input_requests):
        return None

    records: list[dict[str, Any]] = []
    for request, output in zip(input_requests, outputs):
        session = request.sovbench_session
        predictions = list(session.turn_predictions)
        for turn in session.turns:
            prediction = predictions[turn.session_turn_id] if turn.session_turn_id < len(predictions) else ""
            pred_letter = extract_sovbench_choice_letter(prediction)
            records.append(
                {
                    "session_id": session.session_id,
                    "session_turn_id": turn.session_turn_id,
                    "session_num_turns": session.num_turns,
                    "group_id": turn.group_id,
                    "group_num_turns": turn.group_num_turns,
                    "turn_type": turn.turn_type,
                    "answer": turn.answer,
                    "prediction": prediction,
                    "pred_letter": pred_letter,
                    "score": -1 if not pred_letter else int(pred_letter == turn.answer.strip().upper()),
                }
            )

    result: dict[str, Any] = {
        "sovbench_o_accuracy": _mean([r["score"] for r in records]),
        "sovbench_o_correct": sum(int(r["score"]) for r in records if r["score"] >= 0),
        "sovbench_o_total": len(records),
        "sovbench_o_valid": sum(1 for r in records if r["score"] >= 0),
        "sovbench_o_by_turn_type": _category_stats(records, "turn_type"),
        "sovbench_o_by_group_size": _category_stats(records, "group_num_turns"),
        "sovbench_o_by_session_length": _category_stats(records, "session_num_turns"),
    }
    if include_per_item:
        result["sovbench_o_eval_items"] = records
    return result


def print_sovbench_o_accuracy_summary(metrics: dict[str, Any]) -> None:
    """Print the SOVBench-O accuracy summary in the benchmark's result log."""
    accuracy = metrics.get("sovbench_o_accuracy")
    accuracy_text = "nan" if accuracy != accuracy else f"{accuracy:.4f}"
    print("\nSOVBench-O accuracy:")
    print(f"  overall: {accuracy_text} (valid {metrics.get('sovbench_o_valid')}/{metrics.get('sovbench_o_total')})")
    for label, key in (
        ("by turn type", "sovbench_o_by_turn_type"),
        ("by group size", "sovbench_o_by_group_size"),
        ("by session length", "sovbench_o_by_session_length"),
    ):
        breakdown = metrics.get(key)
        if breakdown:
            pairs = ", ".join(
                (
                    f"{value}: acc={stats['accuracy']:.4f} correct={stats['correct']}/{stats['valid']}"
                    f" total={stats['total']}"
                    if stats["accuracy"] is not None
                    else f"{value}: no parseable predictions (total={stats['total']})"
                )
                for value, stats in breakdown.items()
            )
            print(f"  {label}: {pairs}")
