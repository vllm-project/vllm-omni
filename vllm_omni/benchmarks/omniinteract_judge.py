# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Text-only LLM judge for OmniInteract unified evaluation.

The three prompt templates follow the official English judge protocol in
Lucky-Lance/OmniInteract ``eval/evaluation/llm_judge.py``
(commit de304cef35fd9a50a5caadb5090c34cfbf0dd868). They correspond to the
OmniInteract paper appendix (arXiv:2605.26485) Listing A.1 (early-stage),
Listing A.2 (interrupted partial quality), and Listing A.3 (core-stage).
A local OpenAI-compatible judge is not the paper's GPT-4o judge, so scores
are protocol-compatible rather than official paper-table numbers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

import regex as re
import requests

# Official English templates from OmniInteract llm_judge.py.
# Paper appendix: Listing A.1 early, A.2 interrupted-partial, A.3 core.
# Source: Lucky-Lance/OmniInteract@de304cef35fd9a50a5caadb5090c34cfbf0dd868

EARLY_SYSTEM_PROMPT = (
    "You are a streaming voice assistant evaluation judge. "
    "Judge only based on the given text. "
    "Output must be parseable JSON with no other text."
)

EARLY_USER_TEMPLATE = """Determine whether the early output between start and t_a is an early hallucination.

[scene_type] {scene_type}
[slot] slot_id={slot_id},
  turn_index={turn_index},
  step_index={step_index},
  boundary_type={boundary_type},
  is_interrupted={is_interrupted}
[question] {question}
[current_gt_answer] {gt_answer}
[full_chunk_context] {full_context}
[early_actual_text] {actual_text}

Rules:
1. Greetings, confirmations, waiting, brief observations, and follow-up phrases -> Neutral.
2. If the model starts substantively answering, guessing unseen info, revealing
   future steps, or making definitive factual claims -> FP.
3. For 1QnA first step, reciting the full procedure before acting -> FP.
4. score is interaction quality 0-1 when Neutral; 0 when hallucination.

Output JSON:
{{"flag":"Neutral|FP_Hallucination",
 "score":float 0-1,
 "rationale":"one sentence"}}"""

CORE_SYSTEM_PROMPT = (
    "You are a strict streaming voice assistant core-answer evaluation judge. "
    "Judge only based on the given text and reference answer. "
    "Output must be parseable JSON with no other text."
)

CORE_USER_TEMPLATE = """Score the core output after t_a.

[scene_type] {scene_type}
[slot] slot_id={slot_id},
  turn_index={turn_index},
  step_index={step_index},
  boundary_type={boundary_type},
  is_interrupted={is_interrupted}
[question] {question}
[current_gt_answer] {gt_answer}
[future_gt_answers_or_steps]
  {future_answers}
[full_chunk_context] {full_context}
[core_actual_text] {actual_text}

Rules:
1. score 0-1: correctness and coverage of core_actual_text vs gt_answer.
2. Off-topic, factual errors, or missing key answer -> low score.
3. 1QnA: reward only current-step info; penalize spoiling future steps or skipping the current step.
4. If score > 0, extract the earliest contiguous substring from core_actual_text
   that establishes the answer as trigger_phrase.
5. trigger_phrase must be a verbatim substring; empty if score == 0.

Output JSON:
{{"score":float 0-1,
 "trigger_phrase":"substring or empty",
 "spoiler":true|false,
 "rationale":"one sentence"}}"""

INTERRUPT_PARTIAL_SYSTEM_PROMPT = (
    "You are a strict evaluator for interrupted voice-assistant answers. "
    "Judge only from the provided text. "
    "Return valid JSON only, with no extra text."
)

INTERRUPT_PARTIAL_USER_TEMPLATE = """Evaluate the quality of the assistant output
that was already spoken before or around an interruption.

[Task]
The assistant was answering, but the interaction was interrupted before
completion. The assistant was not required to complete the full original answer.
Score whether the content already spoken is relevant, correct, and useful for
the current ground-truth answer.

[Question] {question}
[Ground Truth Answer] {gt_answer}
[Assistant Output Already Spoken]
{actual_text}

Scoring Rules:
1. Score from 0 to 1.
2. Do not penalize incompleteness: a partial answer can receive a high score if the spoken part is correct and useful.
3. Score high when the spoken content overlaps with, paraphrases, or conveys useful parts of the ground truth.
4. Score low for acknowledgments or prefaces without substantive answer content.
5. Score low for wrong-question, irrelevant, or generic-filler output.
6. hallucination=true if the output contains clear incorrect facts, wrong target content, or unsupported content.
7. Ignore overflow duration when scoring quality; spill is measured separately.

Output JSON:
{{"score":float 0-1,
 "hallucination":true|false,
 "rationale":"one sentence"}}"""


class JudgeRequestError(RuntimeError):
    """Raised when the configured judge endpoint cannot produce a response."""


@dataclass(frozen=True)
class EarlyJudgment:
    category: str
    score: float
    rationale: str
    raw: str
    parse_source: str


@dataclass(frozen=True)
class CoreJudgment:
    score: float
    trigger_phrase: str
    spoiler: bool
    rationale: str
    raw: str
    parse_source: str


@dataclass(frozen=True)
class PartialJudgment:
    score: float
    hallucination: bool
    rationale: str
    raw: str
    parse_source: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _text(value: object) -> str:
    return str(value or "").strip()


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "y", "t"}


def _first_json_object(text: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    position = text.find("{")
    while position >= 0:
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            position = text.find("{", position + 1)
            continue
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return value
        position = text.find("{", position + 1)
    return None


def _first_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class OmniInteractJudge:
    """OpenAI-compatible client implementing the OmniInteract judge contract."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        max_tokens: int = 512,
    ) -> None:
        if not base_url.strip():
            raise ValueError("judge base URL is required")
        if not model.strip():
            raise ValueError("judge model is required")
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = requests.post(
                _chat_url(self.base_url),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "stream": False,
                    "max_tokens": self.max_tokens,
                },
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise JudgeRequestError(f"judge request failed: {exc}") from exc
        if not response.ok:
            raise JudgeRequestError(f"judge returned HTTP {response.status_code}: {response.text[:500]}")
        try:
            payload = response.json()
            choices = payload["choices"]
            message = choices[0]["message"]
            return str(message["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise JudgeRequestError(f"unexpected judge response: {response.text[:500]}") from exc

    def judge_early(
        self,
        slot: Mapping[str, object],
        full_context: str,
        actual_text: str,
    ) -> EarlyJudgment:
        raw = self._generate(
            EARLY_SYSTEM_PROMPT,
            EARLY_USER_TEMPLATE.format(
                scene_type=_text(slot.get("scene_type")),
                slot_id=slot.get("slot_id"),
                turn_index=slot.get("turn_index"),
                step_index=slot.get("step_index"),
                boundary_type=_text(slot.get("boundary_type")),
                is_interrupted=str(_bool(slot.get("is_interrupted"))).lower(),
                question=_text(slot.get("question_text")),
                gt_answer=_text(slot.get("gt_answer")),
                full_context=full_context.strip(),
                actual_text=actual_text.strip(),
            ),
        )
        parsed = _first_json_object(raw)
        flag = _text(parsed.get("flag")) if parsed else raw.strip()
        normalized = flag.lower().replace("-", "_").replace(" ", "")
        if "hallucination" in normalized:
            category, score = "hallucination", 0.0
        elif "neutral" in normalized:
            try:
                score = _clamp(float(parsed.get("score", 1.0))) if parsed else 0.0
            except (TypeError, ValueError):
                score = 0.0
            category = "neutral"
        else:
            category, score = "neutral", 0.0
        return EarlyJudgment(
            category=category,
            score=score,
            rationale=_text(parsed.get("rationale")) if parsed else "",
            raw=raw,
            parse_source="llm_json" if parsed else "llm_parse_failed",
        )

    def judge_core(
        self,
        slot: Mapping[str, object],
        full_context: str,
        actual_text: str,
        future_answers: str,
    ) -> CoreJudgment:
        raw = self._generate(
            CORE_SYSTEM_PROMPT,
            CORE_USER_TEMPLATE.format(
                scene_type=_text(slot.get("scene_type")),
                slot_id=slot.get("slot_id"),
                turn_index=slot.get("turn_index"),
                step_index=slot.get("step_index"),
                boundary_type=_text(slot.get("boundary_type")),
                is_interrupted=str(_bool(slot.get("is_interrupted"))).lower(),
                question=_text(slot.get("question_text")),
                gt_answer=_text(slot.get("gt_answer")),
                future_answers=future_answers.strip(),
                full_context=full_context.strip(),
                actual_text=actual_text.strip(),
            ),
        )
        parsed = _first_json_object(raw)
        score: float | None = None
        if parsed is not None:
            try:
                score = _clamp(float(parsed.get("score", 0.0)))
            except (TypeError, ValueError):
                score = None
        fallback = _first_float(raw) if score is None else None
        return CoreJudgment(
            score=score if score is not None else _clamp(fallback) if fallback is not None else 0.0,
            trigger_phrase=_text(parsed.get("trigger_phrase")) if parsed else "",
            spoiler=_bool(parsed.get("spoiler")) if parsed else False,
            rationale=_text(parsed.get("rationale")) if parsed else "",
            raw=raw,
            parse_source="llm_json"
            if score is not None
            else "llm_float_text"
            if fallback is not None
            else "llm_parse_failed",
        )

    def judge_interrupted_partial(
        self,
        slot: Mapping[str, object],
        actual_text: str,
    ) -> PartialJudgment:
        raw = self._generate(
            INTERRUPT_PARTIAL_SYSTEM_PROMPT,
            INTERRUPT_PARTIAL_USER_TEMPLATE.format(
                question=_text(slot.get("question_text")),
                gt_answer=_text(slot.get("gt_answer")),
                actual_text=actual_text.strip(),
            ),
        )
        parsed = _first_json_object(raw)
        try:
            score = _clamp(float(parsed.get("score", 0.0))) if parsed else 0.0
        except (TypeError, ValueError):
            score = 0.0
        return PartialJudgment(
            score=score,
            hallucination=_bool(parsed.get("hallucination")) if parsed else False,
            rationale=_text(parsed.get("rationale")) if parsed else "",
            raw=raw,
            parse_source="llm_json" if parsed else "llm_parse_failed",
        )
