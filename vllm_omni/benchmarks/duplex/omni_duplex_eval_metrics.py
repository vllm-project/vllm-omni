# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Protocol prompts, windows, parsing, and aggregation.

Prompt wording and window math are copied verbatim from the MIT-licensed
OpenBMB/Omni-DuplexEval evaluator at :data:`PROTOCOL_PIN`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import regex as re

# Copied from OpenBMB/Omni-DuplexEval@PROTOCOL_PIN (MIT).
# Do not change prompt wording without bumping the pin and recording a score delta.
PROTOCOL_PIN = "ca3c122b4d4bf67afd6b18ea5e724b4561bdde48"


def temporal_window(sentence_start: float, sentence_end: float, video_duration: float) -> tuple[float, float] | None:
    start = max(0.0, float(sentence_start) - 2.0)
    end = max(start + 0.5, float(sentence_end) - 2.0)
    return (start, end) if end <= float(video_duration) and end - start >= 0.5 else None


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
        patterns = [
            (r'"temporal_score"\s*:\s*(\d+)', "temporal_score", int),
            (r'"is_relevant"\s*:\s*(\d+)', "is_relevant", int),
            (r'"success_score"\s*:\s*([01])', "success_score", int),
            (r'"content_score"\s*:\s*(\d+(?:\.\d+)?)', "content_score", float),
        ]
        for pattern, key, caster in patterns:
            if match := re.search(pattern, str(text)):
                parsed = caster(match.group(1))
                value[key] = round(parsed, 2) if caster is float else parsed
        for pattern, key in [
            (r'"temporal_reasoning"\s*:\s*"([^"]*)"', "temporal_reasoning"),
            (r'"content_reasoning"\s*:\s*"([^"]*)"', "content_reasoning"),
            (r'"reasoning"\s*:\s*"([^"]*)"', "reasoning"),
        ]:
            if match := re.search(pattern, str(text), re.S):
                value[key] = match.group(1).strip()
                break
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
    irrelevant = [row for row in valid if int(row.get("is_relevant", 0)) == 0]
    scores = [int(row.get("temporal_score", 0)) for row in relevant]
    total_valid_duration = sum(float(row.get("sentence_duration", 0.0)) for row in valid)
    irrelevant_duration = sum(float(row.get("sentence_duration", 0.0)) for row in irrelevant)
    return {
        "avg_temporal_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "total_sentences": len(rows),
        "evaluated_sentences": len(valid),
        "error_count": len(rows) - len(valid),
        "relevant_sentences_count": len(relevant),
        "irrelevant_sentences_count": len(irrelevant),
        "score_distribution": {
            "3_points": scores.count(3),
            "2_points": scores.count(2),
            "1_point": scores.count(1),
            "0_points": scores.count(0),
        },
        "relevance_stats": {
            "first_relevant_time": min((row.get("sentence_start") for row in relevant), default=None),
            "total_relevant_duration": round(sum(float(row.get("sentence_duration", 0.0)) for row in relevant), 4),
            "total_irrelevant_duration": round(irrelevant_duration, 4),
            "irrelevant_duration_ratio": round(
                irrelevant_duration / total_valid_duration if total_valid_duration > 0 else 0.0, 4
            ),
        },
        "excluded_irrelevant_sentences": [
            {
                "sentence": row.get("sentence", ""),
                "start": row.get("sentence_start"),
                "end": row.get("sentence_end"),
                "temporal_score": row.get("temporal_score", 0),
            }
            for row in irrelevant
        ],
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
    return f"""
You are evaluating a real-time video description system.

Basic information:
- Video segment: {start_time:.2f}s to {end_time:.2f}s.
- User instruction: {question}
- Model response sentence: "{response_text}"

Evaluation principles:
- Judge only the current video segment.
- Ignore response content that clearly belongs to earlier or later segments unless it contradicts the current segment.
- Language does not matter; evaluate semantic correctness and temporal alignment only.
- Decide whether the sentence is a substantive response or only a filler/polite phrase.

Temporal sensitivity score:
- 3: Excellent temporal alignment. The sentence accurately describes the current segment and follows the instruction.
- 2: Mostly aligned. Minor inaccuracies, omissions, or harmless references to nearby segments are allowed.
- 1: Poor alignment. Major inaccuracies, wrong timing, or mostly irrelevant content.
- 0: No meaningful alignment with the current segment.

Relevance label:
- 1: The sentence contains substantive task- or video-related information.
- 0: The sentence is only filler, acknowledgement, meta-commentary, or generic text without useful content.

Return exactly this JSON object:
{{
  "temporal_score": <0, 1, 2, or 3>,
  "temporal_reasoning": "<brief explanation>",
  "is_relevant": <0 or 1>
}}
"""


def build_content_prompt(response_text: str, question: str, references: list[str] | None = None) -> str:
    reference_block = ""
    if references:
        reference_block = "Reference annotations, for reference only:\n"
        for index, reference in enumerate(references, 1):
            reference_block += f"{index}. {reference}\n"

    return f"""
You are a precise content-accuracy evaluator for video description.

Basic information:
- User instruction: {question}
- Model response: "{response_text}"
{reference_block}

Evaluation goal:
Judge whether the model response is factually accurate for the whole video and aligned with the instruction.
Consider object/action/color/count/spatial/event errors, hallucinations, omissions, and irrelevant content.
Reference annotations are optional guidance; the primary evidence is the video itself.

Scoring:
- Use a decimal score from 0.00 to 3.00.
- Start from 3.00 and deduct for each error.
- Use 0.00 only when the response is empty, completely irrelevant, or contains no correct video facts.
- Output exactly two decimal places.

Return exactly this JSON object:
{{
  "content_score": <decimal from 0.00 to 3.00>,
  "content_reasoning": "<brief explanation with main errors and final score>"
}}
"""


def build_reminder_prompt(instruction: str, response: str, task_type: str, ground_answer: str = "") -> str:
    if task_type == "correction":
        return f"""
You are judging whether a model successfully completed a correction task.

Task description:
The user instruction contains an incorrect statement about the video.
The model should identify the incorrect part and provide the correct information.

Input:
- User instruction: {instruction}
- Reference correction: {ground_answer}
- Model response: "{response}"

Success criteria:
1. Identify the error implied by the user instruction and the reference correction.
2. Check whether the model corrected all required error points.
3. The corrected content must preserve the correct context, including subject, object, action, and attributes.
4. Ignore extra information that is unrelated to both the instruction and the reference, unless it contradicts them.

Scoring:
- 1 = all required error points are corrected and the context is consistent.
- 0 = at least one required correction is missing, wrong, or contextually inconsistent.

Return exactly this JSON object:
{{
  "success_score": <0 or 1>,
  "reasoning": "<brief explanation covering the required correction points>"
}}
"""

    if task_type == "post_event_reminder":
        timing_note = "The response is evaluated immediately after the target event has occurred."
    else:
        timing_note = "The response is evaluated at the target event time or immediately after it."

    return f"""
You are judging whether a model successfully completed an event-reminder task.

Task description:
The user gave an instruction asking the system to remind them when a specific event happens.
{timing_note}
The provided model text is the response segment generated in the evaluation window.

Input:
- User instruction: {instruction}
- Model response segment: "{response}"

Success criteria:
1. The response clearly refers to the target event in the instruction.
2. The response communicates a reminder, notification, or confirmation that the event has happened.
3. Vague narration, unrelated descriptions, or wrong-event reminders are failures.
4. The response does not need to match the instruction wording exactly if the meaning is clear.

Scoring:
- 1 = successful reminder.
- 0 = unsuccessful reminder.

Return exactly this JSON object:
{{
  "success_score": <0 or 1>,
  "reasoning": "<brief explanation>"
}}
"""
