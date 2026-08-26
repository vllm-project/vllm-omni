# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Protocol-compatible prompts, windows, parsing, and aggregation.

Prompts and window math are adapted from OpenBMB/Omni-DuplexEval under MIT.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import regex as re

PROTOCOL_PIN = "962cec448ac37a377ffb963b476778777d11d346"


def temporal_window(sentence_start: float, sentence_end: float, video_duration: float) -> tuple[float, float] | None:
    start = float(sentence_start) - 2.0
    end = float(sentence_end) - 2.0
    return (start, end) if start >= 0.0 and end <= float(video_duration) and end - start >= 0.5 else None


def reminder_window(start_time: float, window_size: float = 10.0) -> tuple[float, float]:
    return float(start_time), float(start_time) + float(window_size)


def parse_judge_json(text: str) -> dict[str, Any]:
    for match in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", str(text), re.S):
        try:
            value = json.loads(re.sub(r",\s*([}\]])", r"\1", match))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            break
    else:
        value = {}
    for key in ("temporal_score", "is_relevant", "success_score"):
        if key in value:
            try:
                value[key] = int(float(value[key]))
            except (TypeError, ValueError):
                value[key] = 0
    if "content_score" in value:
        try:
            value["content_score"] = round(float(value["content_score"]), 2)
        except (TypeError, ValueError):
            value["content_score"] = 0.0
    return value


def summarize_temporal_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    valid = [row for row in rows if not row.get("error")]
    relevant = [row for row in valid if int(row.get("is_relevant", 0)) == 1]
    scores = [int(row.get("temporal_score", 0)) for row in relevant]
    return {
        "avg_temporal_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "total_sentences": len(rows),
        "evaluated_sentences": len(valid),
        "error_count": len(rows) - len(valid),
        "relevant_sentences_count": len(relevant),
        "irrelevant_sentences_count": len(valid) - len(relevant),
        "score_distribution": {f"{score}_points": scores.count(score) for score in range(4)},
    }


def summarize_pr_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    successes = [int(row.get("all_success", row.get("total_score", 0))) for row in rows]
    by_task: dict[str, list[int]] = {}
    for row, score in zip(rows, successes, strict=False):
        by_task.setdefault(str(row.get("task_type", "unknown")), []).append(score)
    return {
        "samples": len(rows),
        "mean_all_success": sum(successes) / len(successes) if successes else 0.0,
        "by_task": {task: sum(values) / len(values) for task, values in by_task.items()},
    }


def build_temporal_prompt(start_time: float, end_time: float, response_text: str, question: str) -> str:
    return f'''You are evaluating a real-time video description system.
Video segment: {start_time:.2f}s to {end_time:.2f}s.
User instruction: {question}
Model response sentence: "{response_text}"
Score temporal alignment from 0 to 3 and mark substantive relevance (0/1).
Return exactly JSON: {{"temporal_score": <0-3>, "temporal_reasoning": "...", "is_relevant": <0-1>}}'''


def build_content_prompt(response_text: str, question: str, references: list[str] | None = None) -> str:
    refs = "\n".join(f"{i}. {item}" for i, item in enumerate(references or [], 1))
    reference_block = f"Reference annotations:\n{refs}" if refs else ""
    return f'''You are a precise content-accuracy evaluator for video description.
User instruction: {question}
Model response: "{response_text}"
{reference_block}
Start from 3.00 and deduct for factual errors. Return exactly JSON:
{{"content_score": <decimal from 0.00 to 3.00>, "content_reasoning": "..."}}'''


def build_reminder_prompt(instruction: str, response: str, task_type: str, ground_answer: str = "") -> str:
    if task_type == "correction":
        criteria = f"Reference correction: {ground_answer}\nIdentify and correct all required errors."
    else:
        criteria = "The response must clearly refer to the target event and communicate a reminder or confirmation."
    return f'''You are judging a {task_type} task.
Instruction: {instruction}
Response segment: "{response}"
{criteria}
Return exactly JSON: {{"success_score": <0 or 1>, "reasoning": "..."}}'''
