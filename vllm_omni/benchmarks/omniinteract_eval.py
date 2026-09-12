# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniInteract slot matching, judge scoring, and batch reporting."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import (
    OmniInteractCase,
    OmniInteractEvaluationOptions,
)
from vllm_omni.benchmarks.omniinteract import OmniInteractCaseResult
from vllm_omni.benchmarks.omniinteract_judge import (
    CoreJudgment,
    EarlyJudgment,
    JudgeRequestError,
    OmniInteractJudge,
    PartialJudgment,
)

HARD = "Hard"
SOFT = "Soft"
PROTOCOL_SOURCE = "Lucky-Lance/OmniInteract@de304cef35fd9a50a5caadb5090c34cfbf0dd868"
_HASH_CHUNK_BYTES = 65536
_TRANSCRIPT_NAME = "wav_transcript.json"


@dataclass(frozen=True)
class AlignedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptChunk:
    source_id: int
    text: str
    start: float
    end: float
    aligned_words: tuple[AlignedWord, ...] = ()


@dataclass(frozen=True)
class Slot:
    slot_id: int
    start: float
    answer_time: float
    end: float
    boundary_type: str
    question_text: str
    gt_answer: str
    scene_type: str
    step_index: int | None = None
    turn_index: int | None = None
    question_type: str = "unknown"
    is_interrupted: bool = False
    label: str = ""
    nested_group_id: int | None = None
    nested_role: str | None = None

    def judge_context(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "scene_type": self.scene_type,
            "step_index": self.step_index,
            "turn_index": self.turn_index,
            "boundary_type": self.boundary_type,
            "is_interrupted": self.is_interrupted,
            "question_text": self.question_text,
            "gt_answer": self.gt_answer,
        }


@dataclass(frozen=True)
class MatchedChunk:
    source_id: int
    text: str
    full_text: str
    start: float
    end: float
    aligned_words: tuple[AlignedWord, ...] = ()
    effective_start_hint: float | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_chunk_id": self.source_id,
            "text": self.text,
            "full_text": self.full_text,
            "start": self.start,
            "end": self.end,
            "aligned_words": [asdict(word) for word in self.aligned_words],
        }
        if self.effective_start_hint is not None:
            payload["effective_start_hint"] = self.effective_start_hint
        return payload


@dataclass
class MatchedSlot:
    slot: Slot
    all_chunks: list[MatchedChunk]
    early_chunks: list[MatchedChunk]
    core_chunks: list[MatchedChunk]


@dataclass(frozen=True)
class EvaluationConfig:
    w_ack: float = 0.2
    decay_alpha: float = 1.0
    decay_gamma: float = 1.0
    core_quality_threshold: float = 0.5
    partial_quality_threshold: float = 0.5
    last_slot_tail_s: float = 60.0
    count_unmatched_as_fp: bool = True


class Judge(Protocol):
    def judge_early(
        self,
        slot: dict[str, object],
        full_context: str,
        actual_text: str,
    ) -> EarlyJudgment: ...

    def judge_core(
        self,
        slot: dict[str, object],
        full_context: str,
        actual_text: str,
        future_answers: str,
    ) -> CoreJudgment: ...

    def judge_interrupted_partial(
        self,
        slot: dict[str, object],
        actual_text: str,
    ) -> PartialJudgment: ...


@dataclass(frozen=True)
class EvaluationInputsFingerprint:
    """Hashes and judge identity used to decide whether a cached eval is reusable."""

    annotation_sha256: str
    transcript_sha256: str
    judge_model: str
    judge_base_url: str
    judge_max_tokens: int
    protocol_source: str

    @classmethod
    def from_mapping(cls, payload: object) -> EvaluationInputsFingerprint | None:
        if not isinstance(payload, Mapping):
            return None
        try:
            return cls(
                annotation_sha256=str(payload["annotation_sha256"]),
                transcript_sha256=str(payload["transcript_sha256"]),
                judge_model=str(payload["judge_model"]),
                judge_base_url=_canonical_judge_base_url(str(payload["judge_base_url"])),
                judge_max_tokens=int(payload["judge_max_tokens"]),
                protocol_source=str(payload["protocol_source"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _canonical_judge_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_inputs_fingerprint(
    *,
    annotation_path: Path,
    transcript_path: Path,
    judge_model: str,
    judge_base_url: str,
    judge_max_tokens: int,
) -> EvaluationInputsFingerprint:
    """Fingerprint transcript, annotation, and judge configuration for cache reuse."""

    return EvaluationInputsFingerprint(
        annotation_sha256=_sha256_file(annotation_path),
        transcript_sha256=_sha256_file(transcript_path),
        judge_model=judge_model,
        judge_base_url=_canonical_judge_base_url(judge_base_url),
        judge_max_tokens=judge_max_tokens,
        protocol_source=PROTOCOL_SOURCE,
    )


def _fingerprint_from_options(
    case: OmniInteractCase,
    result: OmniInteractCaseResult,
    options: OmniInteractEvaluationOptions,
) -> EvaluationInputsFingerprint:
    return evaluation_inputs_fingerprint(
        annotation_path=case.annotation_path,
        transcript_path=Path(result.output_dir) / _TRANSCRIPT_NAME,
        judge_model=options.judge_model,
        judge_base_url=options.judge_base_url,
        judge_max_tokens=options.judge_max_tokens,
    )


@dataclass(frozen=True)
class _QaRow:
    question_time: float
    answer_time: float
    question_text: str
    answer_text: str
    question_type: str
    is_interrupted: bool
    label: str = ""


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return value


def _items(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _text(value: object) -> str:
    return str(value or "").strip()


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "y", "t"}


def _time(value: object) -> float | None:
    if isinstance(value, int | float):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        pass
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        return numbers[0] * 60.0 + numbers[1]
    return numbers[0] * 3600.0 + numbers[1] * 60.0 + numbers[2]


def _qa_rows(root: object) -> list[_QaRow]:
    values = root if isinstance(root, list) else _mapping(root).get("items", [])
    rows = []
    for value in _items(values):
        if not isinstance(value, dict):
            continue
        question_time = _time(value.get("question_time"))
        answer_time = _time(value.get("answer_time"))
        if question_time is None or answer_time is None:
            continue
        rows.append(
            _QaRow(
                question_time=question_time,
                answer_time=answer_time,
                question_text=_text(value.get("question_text")),
                answer_text=_text(value.get("answer_text")),
                question_type=_text(value.get("question_type")).lower() or "unknown",
                is_interrupted=_bool(value.get("is_interrupted")),
            )
        )
    return sorted(rows, key=lambda row: (row.question_time, row.answer_time))


def _multi_turn_slots(root: object, tail_s: float, scene_type: str) -> list[Slot]:
    rows = _qa_rows(root)
    return [
        Slot(
            slot_id=index + 1,
            start=row.question_time,
            answer_time=row.answer_time,
            end=max(
                row.question_time,
                rows[index + 1].question_time if index + 1 < len(rows) else row.answer_time + tail_s,
            ),
            boundary_type=HARD,
            question_text=row.question_text,
            gt_answer=row.answer_text,
            scene_type=scene_type,
            turn_index=index + 1,
            question_type=row.question_type,
            is_interrupted=row.is_interrupted,
        )
        for index, row in enumerate(rows)
    ]


def _one_to_many_slots(root: object, tail_s: float) -> list[Slot]:
    data = _mapping(root)
    question_time = _time(data.get("question_time")) or 0.0
    question_text = _text(data.get("question_text"))
    inferred = _text(data.get("inferred_knowledge"))
    answers: list[_QaRow] = []
    conversations = _items(data.get("conversations"))
    if conversations:
        for value in conversations:
            if not isinstance(value, dict):
                continue
            role = _text(value.get("from"))
            if role == "user":
                question_time = _time(value.get("timestamp")) or question_time
                question_text = _text(value.get("value")) or question_text
            elif role == "assistant":
                answer_time = _time(value.get("timestamp"))
                if answer_time is not None:
                    answers.append(
                        _QaRow(
                            question_time=question_time,
                            answer_time=answer_time,
                            question_text=question_text,
                            answer_text=_text(value.get("value")),
                            question_type=_text(value.get("label")).lower() or "step",
                            is_interrupted=_bool(value.get("interrupted")),
                            label=_text(value.get("label")),
                        )
                    )
    else:
        for value in _items(data.get("answers")):
            if not isinstance(value, dict):
                continue
            answer_time = _time(value.get("answer_time"))
            if answer_time is not None:
                answers.append(
                    _QaRow(
                        question_time=question_time,
                        answer_time=answer_time,
                        question_text=question_text,
                        answer_text=_text(value.get("answer_text")),
                        question_type=_text(value.get("label")).lower() or "step",
                        is_interrupted=_bool(value.get("interrupted", value.get("is_interrupted"))),
                        label=_text(value.get("label")),
                    )
                )
    if not conversations and "answers" not in data:
        raise ValueError("1QnA annotation must contain conversations[] or answers[]")
    answers.sort(key=lambda row: row.answer_time)
    if conversations and inferred:
        question_text = f"{inferred}\n\n{question_text}".strip()
    return [
        Slot(
            slot_id=index + 1,
            start=question_time if index == 0 else row.answer_time,
            answer_time=row.answer_time,
            end=max(
                row.answer_time,
                answers[index + 1].answer_time if index + 1 < len(answers) else row.answer_time + tail_s,
            ),
            boundary_type=SOFT,
            question_text=question_text,
            gt_answer=row.answer_text,
            scene_type="1QnA",
            step_index=index + 1,
            turn_index=1,
            question_type=row.question_type,
            is_interrupted=row.is_interrupted,
            label=row.label,
        )
        for index, row in enumerate(answers)
    ]


def _nested_slots(root: object, tail_s: float) -> list[Slot]:
    rows = _qa_rows(root)
    groups: list[tuple[_QaRow, _QaRow]] = []
    index = 0
    while index < len(rows):
        outer = rows[index]
        inner_index = next(
            (
                candidate
                for candidate in range(index + 1, len(rows))
                if outer.question_time < rows[candidate].question_time < outer.answer_time
                and rows[candidate].answer_time <= outer.answer_time
            ),
            None,
        )
        if inner_index is None:
            raise ValueError("failed to infer nested question pair")
        groups.append((outer, rows[inner_index]))
        index = inner_index + 1
    slots = []
    for group_index, (outer, inner) in enumerate(groups, start=1):
        next_question = (
            groups[group_index][0].question_time if group_index < len(groups) else outer.answer_time + tail_s
        )
        slots.extend(
            [
                Slot(
                    slot_id=len(slots) + 1,
                    start=outer.question_time,
                    answer_time=outer.answer_time,
                    end=max(next_question, outer.answer_time),
                    boundary_type=HARD,
                    question_text=outer.question_text,
                    gt_answer=outer.answer_text,
                    scene_type="nested",
                    turn_index=group_index,
                    question_type=outer.question_type,
                    is_interrupted=outer.is_interrupted,
                    nested_group_id=group_index,
                    nested_role="outer",
                ),
                Slot(
                    slot_id=len(slots) + 2,
                    start=inner.question_time,
                    answer_time=inner.answer_time,
                    end=outer.answer_time,
                    boundary_type=HARD,
                    question_text=inner.question_text,
                    gt_answer=inner.answer_text,
                    scene_type="nested",
                    turn_index=group_index,
                    question_type=inner.question_type,
                    is_interrupted=inner.is_interrupted,
                    nested_group_id=group_index,
                    nested_role="inner",
                ),
            ]
        )
    return slots


def build_slots(root: object, scene_type: str, tail_s: float = 60.0) -> list[Slot]:
    """Build official OmniInteract evaluation windows from an annotation."""

    if scene_type == "multi_turn":
        return _multi_turn_slots(root, tail_s, scene_type)
    if scene_type == "nested":
        return _nested_slots(root, tail_s)
    if scene_type == "1QnA":
        return _one_to_many_slots(root, tail_s)
    raise ValueError(f"unsupported OmniInteract scene type: {scene_type!r}")


def load_transcript(path: Path) -> tuple[list[TranscriptChunk], bool]:
    """Load timestamped transcript chunks and report complete word alignment."""

    data = _mapping(_read_json(path))
    chunks = []
    for source_id, value in enumerate(_items(data.get("chunks"))):
        if not isinstance(value, dict):
            continue
        timestamp = value.get("timestamp")
        if not isinstance(timestamp, list) or len(timestamp) != 2:
            continue
        start, end = _time(timestamp[0]), _time(timestamp[1])
        text = _text(value.get("text"))
        if start is None or end is None or not text:
            continue
        words = []
        for raw_word in _items(value.get("aligned_words")):
            if not isinstance(raw_word, dict):
                continue
            word_start, word_end = _time(raw_word.get("start")), _time(raw_word.get("end"))
            word_text = _text(raw_word.get("text"))
            if word_start is not None and word_end is not None and word_text:
                words.append(AlignedWord(word_text, min(word_start, word_end), max(word_start, word_end)))
        chunks.append(
            TranscriptChunk(
                source_id=source_id,
                text=text,
                start=min(start, end),
                end=max(start, end),
                aligned_words=tuple(sorted(words, key=lambda word: (word.start, word.end))),
            )
        )
    chunks.sort(key=lambda chunk: (chunk.start, chunk.end, chunk.source_id))
    fully_aligned = bool(chunks) and all(chunk.aligned_words for chunk in chunks)
    return chunks, fully_aligned


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _join_words(words: tuple[AlignedWord, ...]) -> str:
    separator = "" if _contains_cjk("".join(word.text for word in words)) else " "
    return separator.join(word.text for word in words).strip()


def _matched(
    chunk: TranscriptChunk,
    words: tuple[AlignedWord, ...] | None = None,
    *,
    effective_start_hint: float | None = None,
) -> MatchedChunk:
    chosen = chunk.aligned_words if words is None else words
    return MatchedChunk(
        source_id=chunk.source_id,
        text=chunk.text if words is None else _join_words(chosen),
        full_text=chunk.text,
        start=chosen[0].start if chosen else chunk.start,
        end=chosen[-1].end if chosen else chunk.end,
        aligned_words=chosen,
        effective_start_hint=effective_start_hint,
    )


def _sort_matched(chunks: list[MatchedChunk]) -> list[MatchedChunk]:
    return sorted(chunks, key=lambda chunk: (chunk.start, chunk.end))


def match_slots(slots: list[Slot], chunks: list[TranscriptChunk]) -> tuple[list[MatchedSlot], list[TranscriptChunk]]:
    """Assign transcript chunks to slots and split chunks crossing answer time."""

    rows = [MatchedSlot(slot, [], [], []) for slot in slots]
    unmatched = []
    for chunk in chunks:
        candidates = [index for index, slot in enumerate(slots) if slot.start <= chunk.start < slot.end]
        if not candidates:
            unmatched.append(chunk)
            continue
        index = max(candidates, key=lambda candidate: (slots[candidate].start, slots[candidate].slot_id))
        row, slot = rows[index], slots[index]
        whole = _matched(chunk)
        row.all_chunks.append(whole)
        if chunk.start < slot.answer_time < chunk.end and chunk.aligned_words:
            early_words = tuple(word for word in chunk.aligned_words if word.start < slot.answer_time)
            core_words = tuple(word for word in chunk.aligned_words if word.start >= slot.answer_time)
            if early_words:
                row.early_chunks.append(_matched(chunk, early_words))
            if core_words:
                row.core_chunks.append(_matched(chunk, core_words, effective_start_hint=slot.answer_time))
            if early_words or core_words:
                continue
        target = row.early_chunks if chunk.start < slot.answer_time else row.core_chunks
        target.append(whole)
    for row in rows:
        row.all_chunks = _sort_matched(row.all_chunks)
        row.early_chunks = _sort_matched(row.early_chunks)
        row.core_chunks = _sort_matched(row.core_chunks)
    return rows, unmatched


def _decay(value: float, peak: float, end: float, alpha: float, gamma: float) -> float:
    if end <= peak:
        return 1.0 if value <= peak else 0.0
    if value <= peak:
        return 1.0
    if value >= end:
        return 0.0
    return max(0.0, min(1.0, 1.0 - alpha * (((value - peak) / (end - peak)) ** gamma)))


def _chunk_text(chunks: list[MatchedChunk]) -> str:
    return "".join(chunk.text for chunk in chunks).strip()


def _full_context(stage_chunks: list[MatchedChunk], all_chunks: list[MatchedChunk] | None = None) -> str:
    if not stage_chunks:
        return ""
    wanted = {chunk.source_id for chunk in stage_chunks}
    values: list[str] = []
    seen: set[int] = set()
    for chunk in _sort_matched(all_chunks if all_chunks is not None else stage_chunks):
        if chunk.source_id in wanted and chunk.source_id not in seen and chunk.full_text:
            seen.add(chunk.source_id)
            values.append(chunk.full_text)
    if not values:
        for chunk in _sort_matched(stage_chunks):
            text = chunk.full_text or chunk.text
            if text:
                values.append(text)
    return "".join(values).strip()


def _core_effective_start(chunk: MatchedChunk, answer_time: float) -> float:
    if chunk.effective_start_hint is None:
        return max(chunk.start, answer_time)
    return max(min(chunk.start, chunk.effective_start_hint), answer_time)


def _future_answers(slots: list[Slot], index: int) -> str:
    rows = [
        f"- slot {slot.slot_id}: {slot.gt_answer}"
        for slot in slots[index + 1 :]
        if slot.scene_type == slots[index].scene_type
    ]
    return "\n".join(rows) if rows else "(none)"


def _trigger_start(chunks: list[MatchedChunk], phrase: str) -> float | None:
    phrase = phrase.strip().strip("\"'“”‘’")
    if not phrase:
        return None
    text_parts: list[str] = []
    times: list[float | None] = []
    for chunk in chunks:
        if not chunk.aligned_words:
            text_parts.append(chunk.text)
            times.extend([None] * len(chunk.text))
            continue
        separator = "" if _contains_cjk("".join(word.text for word in chunk.aligned_words)) else " "
        built_text = separator.join(word.text for word in chunk.aligned_words)
        built_times: list[float | None] = []
        for word_index, word in enumerate(chunk.aligned_words):
            built_times.extend([word.start] * len(word.text))
            if separator and word_index + 1 < len(chunk.aligned_words):
                built_times.append(None)
        text_parts.append(built_text)
        times.extend(built_times)
    text = "".join(text_parts)
    position = text.find(phrase)
    if position < 0:
        position = text.lower().find(phrase.lower())
    if position < 0:
        return None
    return next(
        (
            times[index]
            for index in range(position, min(len(times), position + len(phrase)))
            if times[index] is not None
        ),
        None,
    )


def _score_slot(
    matched: MatchedSlot,
    slots: list[Slot],
    slot_index: int,
    judge: Judge,
    config: EvaluationConfig,
) -> dict[str, object]:
    slot = matched.slot
    context = slot.judge_context()
    local_w_ack = 0.0 if slot.scene_type == "1QnA" and (slot.step_index or 1) > 1 else config.w_ack
    fp = 0
    early_text = _chunk_text(matched.early_chunks)
    if matched.early_chunks:
        judgment = judge.judge_early(
            context,
            _full_context(matched.early_chunks, matched.all_chunks),
            early_text,
        )
        t_ack = _decay(
            matched.early_chunks[0].start,
            slot.start,
            slot.answer_time,
            config.decay_alpha,
            config.decay_gamma,
        )
        if judgment.category == "hallucination":
            fp += 1
            tp_ack = 0.0
        elif judgment.category == "neutral":
            tp_ack = t_ack * judgment.score * local_w_ack
        else:
            tp_ack = 0.0
        early = {
            "category": judgment.category,
            "actual_text": early_text,
            "T_ack": t_ack,
            "S_ack": judgment.score,
            "W_ack_used": local_w_ack,
            "TP_ack": tp_ack,
            "rationale": judgment.rationale,
            "parse_source": judgment.parse_source,
            "raw": judgment.raw,
        }
    else:
        tp_ack = 0.0
        early = {"category": "none", "actual_text": "", "T_ack": 0.0, "S_ack": 0.0, "TP_ack": 0.0}

    core_text = _chunk_text(matched.core_chunks)
    tp_core = 0.0
    if slot.is_interrupted:
        core: dict[str, object] = {
            "skipped_interrupted": True,
            "actual_text": core_text,
            "T_core": 0.0,
            "S_core": 0.0,
            "TP_core": 0.0,
        }
    elif matched.core_chunks:
        judgment = judge.judge_core(
            context,
            _full_context(matched.core_chunks, matched.all_chunks),
            core_text,
            _future_answers(slots, slot_index),
        )
        fallback_start = _core_effective_start(matched.core_chunks[0], slot.answer_time)
        trigger_start = _trigger_start(matched.core_chunks, judgment.trigger_phrase) if judgment.score > 0 else None
        answer_start = max(trigger_start if trigger_start is not None else fallback_start, slot.answer_time)
        t_core = _decay(
            answer_start,
            slot.answer_time,
            slot.end,
            config.decay_alpha,
            config.decay_gamma,
        )
        low_quality = judgment.score < config.core_quality_threshold
        if low_quality:
            fp += 1
        else:
            tp_core = t_core * judgment.score * (1.0 - local_w_ack)
        core = {
            "skipped_interrupted": False,
            "actual_text": core_text,
            "answer_start": answer_start,
            "trigger_start": trigger_start,
            "trigger_phrase": judgment.trigger_phrase,
            "spoiler": judgment.spoiler,
            "T_core": t_core,
            "S_core": judgment.score,
            "W_core_used": 1.0 - local_w_ack,
            "TP_core": tp_core,
            "low_quality_fp": low_quality,
            "rationale": judgment.rationale,
            "parse_source": judgment.parse_source,
            "raw": judgment.raw,
        }
    else:
        core = {
            "skipped_interrupted": False,
            "actual_text": "",
            "T_core": 0.0,
            "S_core": 0.0,
            "TP_core": 0.0,
        }

    effective_hard_boundary = slot.boundary_type == HARD or slot.is_interrupted
    spill_seconds = (
        max(0.0, max(chunk.end for chunk in matched.all_chunks) - slot.end)
        if effective_hard_boundary and matched.all_chunks
        else 0.0
    )
    if spill_seconds > 0:
        fp += 1
    interruption: dict[str, object] = {}
    if slot.is_interrupted:
        all_text = _chunk_text(matched.all_chunks)
        if not all_text:
            interruption = {"status": "no_output", "has_output": False, "spill_seconds": spill_seconds}
        else:
            partial = judge.judge_interrupted_partial(context, all_text)
            interruption = {
                "status": "ok",
                "has_output": True,
                "partial_quality": partial.score,
                "low_quality": partial.score < config.partial_quality_threshold,
                "hallucination": partial.hallucination,
                "spill_seconds": spill_seconds,
                "TP": partial.score if partial.score >= config.partial_quality_threshold else 0.0,
                "FP": int(partial.score < config.partial_quality_threshold)
                + int(partial.hallucination)
                + int(spill_seconds > 0),
                "FN": int(partial.score < config.partial_quality_threshold),
                "rationale": partial.rationale,
                "parse_source": partial.parse_source,
                "raw": partial.raw,
            }
    fn = 0 if slot.is_interrupted else int(tp_core <= 0)
    return {
        **asdict(slot),
        "t_a": slot.answer_time,
        "num_chunks": len(matched.all_chunks),
        "num_early": len(matched.early_chunks),
        "num_core": len(matched.core_chunks),
        "TP_ack": tp_ack,
        "TP_core": tp_core,
        "Score_ack": tp_ack,
        "Score_core": tp_core,
        "TP_n": max(0.0, min(1.0, tp_ack + tp_core)),
        "FP_delta": fp,
        "FN_delta": fn,
        "stage_early": early,
        "stage_core": core,
        "spill": spill_seconds > 0,
        "spill_seconds": spill_seconds,
        "interruption_diagnostic": interruption,
        "all_chunks": [chunk.as_dict() for chunk in matched.all_chunks],
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _metric(tp: float, fp: int, fn: int, num_slots: int) -> dict[str, object]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "num_slots": num_slots,
        "Global_TP": tp,
        "Global_FP": fp,
        "Global_FN": fn,
        "Precision": precision,
        "Recall": recall,
        "IA_QTF1": _safe_div(2.0 * precision * recall, precision + recall),
    }


def _metric_number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return default


def _metric_sub(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    """Subtract nested inner/outer counts from question-type totals (paper Table 3)."""

    tp = _metric_number(left.get("Global_TP")) - _metric_number(right.get("Global_TP"))
    fp = _metric_number(left.get("Global_FP")) - _metric_number(right.get("Global_FP"))
    fn = _metric_number(left.get("Global_FN")) - _metric_number(right.get("Global_FN"))
    slots = _metric_number(left.get("num_slots")) - _metric_number(right.get("num_slots"))
    return _metric(tp, int(max(0.0, fp)), int(max(0.0, fn)), int(max(0.0, slots)))


def _group_metrics(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_text(row.get(field)) or "unknown"].append(row)
    return {
        name: _metric(
            sum(float(row["TP_n"]) for row in values),
            sum(int(row["FP_delta"]) for row in values),
            sum(int(row["FN_delta"]) for row in values),
            len(values),
        )
        for name, values in sorted(groups.items())
    }


def _nested_by_role(rows: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        role = _text(row.get("nested_role"))
        if role in {"inner", "outer"}:
            groups[role].append(row)
    return {
        name: _metric(
            sum(float(row["TP_n"]) for row in values),
            sum(int(row["FP_delta"]) for row in values),
            sum(int(row["FN_delta"]) for row in values),
            len(values),
        )
        for name, values in sorted(groups.items())
    }


def _empty_metric() -> dict[str, object]:
    return _metric(0.0, 0, 0, 0)


def _as_metric(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else _empty_metric()


def _group_map(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _paper_metrics(summary: dict[str, object]) -> dict[str, object]:
    """Build mutually exclusive Table 3 IA-QTF1 slices from aggregate groups."""

    by_scene_map = _group_map(summary.get("by_scene_type"))
    by_qtype_map = _group_map(summary.get("by_question_type"))
    nested_role_map = _group_map(summary.get("nested_by_role"))
    interruption_raw = summary.get("interruption")
    interruption = interruption_raw if isinstance(interruption_raw, dict) else {}
    nested_raw = summary.get("nested")
    nested = nested_raw if isinstance(nested_raw, dict) else {}

    realtime = _metric_sub(_as_metric(by_qtype_map.get("realtime")), _as_metric(nested_role_map.get("inner")))
    proactive = _metric_sub(
        _as_metric(by_qtype_map.get("proactive")),
        _as_metric(nested_role_map.get("outer")),
    )
    nested_metric = _as_metric(by_scene_map.get("nested"))
    one_qna = _as_metric(by_scene_map.get("1QnA"))

    one_q1a_tp = (
        _metric_number(realtime.get("Global_TP"))
        + _metric_number(proactive.get("Global_TP"))
        + _metric_number(nested_metric.get("Global_TP"))
    )
    one_q1a_fp = (
        _metric_number(realtime.get("Global_FP"))
        + _metric_number(proactive.get("Global_FP"))
        + _metric_number(nested_metric.get("Global_FP"))
    )
    one_q1a_fn = (
        _metric_number(realtime.get("Global_FN"))
        + _metric_number(proactive.get("Global_FN"))
        + _metric_number(nested_metric.get("Global_FN"))
    )
    one_q1a_slots = (
        int(_metric_number(realtime.get("num_slots")))
        + int(_metric_number(proactive.get("num_slots")))
        + int(_metric_number(nested_metric.get("num_slots")))
    )
    return {
        "exp_f1": {
            "realtime": realtime,
            "proactive": proactive,
            "nested": nested_metric,
            "one_q1a_global": _metric(one_q1a_tp, int(one_q1a_fp), int(one_q1a_fn), one_q1a_slots),
            "one_qna": one_qna,
            "all_global": {
                "num_slots": summary.get("num_slots", 0),
                "Global_TP": summary.get("Global_TP", 0.0),
                "Global_FP": summary.get("Global_FP", 0),
                "Global_FN": summary.get("Global_FN", 0),
                "Precision": summary.get("Precision", 0.0),
                "Recall": summary.get("Recall", 0.0),
                "IA_QTF1": summary.get("IA_QTF1", 0.0),
            },
            "definition": (
                "Matches OmniInteract paper Table 3: realtime/proactive exclude nested "
                "inner/outer slots; 1Q1A Global recomputes F1 from those three TP/FP/FN."
            ),
        },
        "exp_interruption": {
            "NOR": interruption.get("NOR", 0.0),
            "PAQ": interruption.get("PAQ", 0.0),
            "CSM_SR": interruption.get("CSM_SR", 0.0),
            "CSM_AS_seconds": interruption.get("CSM_AS", 0.0),
            "interrupted_slot_count": interruption.get("interrupted_slot_count", 0),
        },
        "exp_nested": {
            "NCCS": nested.get("NCCS", 0.0),
            "inner_IA_QTF1": _as_metric(nested_role_map.get("inner")).get("IA_QTF1", 0.0),
            "outer_IA_QTF1": _as_metric(nested_role_map.get("outer")).get("IA_QTF1", 0.0),
            "missed_outer": nested.get("missing_q1_count", 0),
            "num_pairs": nested.get("num_pairs", 0),
            "success_pairs": nested.get("success_pairs", 0),
        },
    }


def _slot_core_score(row: dict[str, object]) -> float:
    if "Score_core" in row:
        return float(row["Score_core"])
    return float(row["TP_core"])


def _nested_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        role = _text(row.get("nested_role"))
        group_id = row.get("nested_group_id")
        if role in {"inner", "outer"} and isinstance(group_id, int):
            groups[(_text(row.get("sample_id")), group_id)][role] = row
    score_sum = 0.0
    success = 0
    missing_q1 = 0
    for pair in groups.values():
        inner, outer = pair.get("inner"), pair.get("outer")
        if inner is None or outer is None:
            continue
        inner_score, outer_score = _slot_core_score(inner), _slot_core_score(outer)
        if outer_score <= 0:
            missing_q1 += 1
        inner_core = inner.get("stage_core")
        outer_core = outer.get("stage_core")
        inner_start = inner_core.get("answer_start") if isinstance(inner_core, dict) else None
        outer_start = outer_core.get("answer_start") if isinstance(outer_core, dict) else None
        # Official NCCS: missing timestamps do not fail the pair on ordering.
        order_ok = True
        if (
            inner_score > 0
            and outer_score > 0
            and isinstance(inner_start, int | float)
            and isinstance(outer_start, int | float)
        ):
            order_ok = float(inner_start) <= float(outer_start)
        if inner_score > 0 and outer_score > 0 and order_ok:
            score_sum += math.sqrt(inner_score * outer_score)
            success += 1
    return {
        "num_pairs": len(groups),
        "success_pairs": success,
        "missing_q1_count": missing_q1,
        "NCCS": _safe_div(score_sum, len(groups)),
    }


def _interruption_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    interrupted = [row for row in rows if bool(row.get("is_interrupted"))]
    with_output = []
    partial_scores = []
    spill_count = 0
    spill_seconds = 0.0
    for row in interrupted:
        diagnostic = row.get("interruption_diagnostic")
        if not isinstance(diagnostic, dict) or not diagnostic.get("has_output"):
            continue
        with_output.append(row)
        if diagnostic.get("status") == "ok":
            partial_scores.append(float(diagnostic.get("partial_quality", 0.0)))
        spill = float(diagnostic.get("spill_seconds", 0.0))
        spill_seconds += spill
        spill_count += int(spill > 0)
    return {
        "interrupted_slot_count": len(interrupted),
        "interrupted_no_output_count": len(interrupted) - len(with_output),
        "NOR": _safe_div(len(interrupted) - len(with_output), len(interrupted)),
        "PAQ": _safe_div(sum(partial_scores), len(partial_scores)),
        "CSM_SR": _safe_div(spill_count, len(with_output)),
        "CSM_AS": _safe_div(spill_seconds, len(with_output)),
    }


def _scenario_case_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    """Count unique evaluated cases per OmniInteract scenario tag."""

    by_tag: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sample_id = _text(row.get("sample_id"))
        if not sample_id:
            continue
        scene = _text(row.get("scene_type"))
        question_type = _text(row.get("question_type")).lower()
        if scene == "1QnA":
            by_tag["1qna"].add(sample_id)
        elif scene == "nested":
            by_tag["nested"].add(sample_id)
        elif question_type == "realtime":
            by_tag["realtime"].add(sample_id)
        elif question_type == "proactive":
            by_tag["proactive"].add(sample_id)
        if bool(row.get("is_interrupted")):
            by_tag["interrupted"].add(sample_id)
    return {tag: len(by_tag.get(tag, ())) for tag in ("realtime", "proactive", "nested", "interrupted", "1qna")}


def _summarize(rows: list[dict[str, object]], unmatched: int) -> dict[str, object]:
    tp = sum(float(row["TP_n"]) for row in rows)
    fp = sum(int(row["FP_delta"]) for row in rows) + unmatched
    fn = sum(int(row["FN_delta"]) for row in rows)
    summary = {
        **_metric(tp, fp, fn, len(rows)),
        "num_unmatched_chunks": unmatched,
        "by_scene_type": _group_metrics(rows, "scene_type"),
        "by_question_type": _group_metrics(rows, "question_type"),
        "nested_by_role": _nested_by_role(rows),
        "interruption": _interruption_summary(rows),
        "nested": _nested_summary(rows),
        "scenario_case_counts": _scenario_case_counts(rows),
    }
    summary["paper_metrics"] = _paper_metrics(summary)
    return summary


def evaluate_case(
    case: OmniInteractCase,
    result: OmniInteractCaseResult,
    judge: Judge,
    output_path: Path,
    config: EvaluationConfig = EvaluationConfig(),
    inputs_fingerprint: EvaluationInputsFingerprint | None = None,
) -> dict[str, object]:
    """Evaluate one published OmniInteract case."""

    annotation = _read_json(case.annotation_path)
    scene_type = "1QnA" if case.scene_type == "1qna" else case.scene_type
    slots = build_slots(annotation, scene_type, config.last_slot_tail_s)
    transcript_path = Path(result.output_dir) / _TRANSCRIPT_NAME
    chunks, fully_aligned = load_transcript(transcript_path)
    matched, unmatched = match_slots(slots, chunks)
    sample_id = f"{case.subset}__{Path(result.output_dir).name}"
    slot_rows = []
    for index, row in enumerate(matched):
        scored = _score_slot(row, slots, index, judge, config)
        scored["sample_id"] = sample_id
        slot_rows.append(scored)
    fingerprint = inputs_fingerprint or evaluation_inputs_fingerprint(
        annotation_path=case.annotation_path,
        transcript_path=transcript_path,
        judge_model=str(getattr(judge, "model", "") or ""),
        judge_base_url=str(getattr(judge, "base_url", "") or ""),
        judge_max_tokens=int(getattr(judge, "max_tokens", 0) or 0),
    )
    evaluation = {
        "status": "ok",
        "sample_id": sample_id,
        "subset": case.subset,
        "scene_type": scene_type,
        "gt_json": str(case.annotation_path.resolve()),
        "model_json": str(transcript_path.resolve()),
        "timing_precision": "word-aligned" if fully_aligned else "chunk-level-approximate",
        "protocol_source": PROTOCOL_SOURCE,
        "inputs_fingerprint": asdict(fingerprint),
        "summary": _summarize(slot_rows, len(unmatched) if config.count_unmatched_as_fp else 0),
        "slots": slot_rows,
        "unmatched_chunks": [asdict(chunk) for chunk in unmatched],
    }
    _write_json(output_path, evaluation)
    return evaluation


_REPORT_WIDTH = 55
_REPORT_LABEL_WIDTH = 26


def _report_rule(fill: str, title: str = "") -> str:
    if not title:
        return fill * _REPORT_WIDTH
    body = f" {title} "
    pad = max(0, _REPORT_WIDTH - len(body))
    left = pad // 2
    return f"{fill * left}{body}{fill * (pad - left)}"


def print_evaluation_report(evaluation: dict[str, object]) -> None:
    """Print a compact accuracy report for the current serving benchmark."""

    summary = evaluation.get("summary")
    if not isinstance(summary, dict):
        return
    interruption = summary.get("interruption")
    nested = summary.get("nested")
    paper = summary.get("paper_metrics")
    interruption = interruption if isinstance(interruption, dict) else {}
    nested = nested if isinstance(nested, dict) else {}
    exp_f1_raw = paper.get("exp_f1") if isinstance(paper, dict) else None
    exp_f1 = exp_f1_raw if isinstance(exp_f1_raw, dict) else {}
    exp_nested_raw = paper.get("exp_nested") if isinstance(paper, dict) else None
    exp_nested = exp_nested_raw if isinstance(exp_nested_raw, dict) else {}

    def _line(label: str, value: object) -> None:
        print(f"{label:<{_REPORT_LABEL_WIDTH}}{value}")

    def _slice_slots(key: str) -> int:
        row = exp_f1.get(key)
        if not isinstance(row, dict):
            return 0
        return int(_metric_number(row.get("num_slots")))

    def _f1_line(label: str, key: str) -> None:
        row = exp_f1.get(key)
        value = row.get("IA_QTF1", 0.0) if isinstance(row, dict) else summary.get("IA_QTF1", 0.0)
        _line(label, f"{_metric_number(value):.6f}")

    print()
    print(_report_rule("=", "OmniInteract Accuracy"))
    _line("Judge model:", evaluation.get("judge_model"))
    _line(
        "Cases:",
        f"{evaluation.get('evaluated', 0)} evaluated, "
        f"{evaluation.get('failed', 0)} failed, "
        f"{evaluation.get('skipped', 0)} skipped",
    )
    counts_raw = summary.get("scenario_case_counts")
    counts = counts_raw if isinstance(counts_raw, dict) else {}
    present_tags = [
        f"{tag} {int(_metric_number(counts.get(tag)))}"
        for tag in ("realtime", "proactive", "nested", "interrupted", "1qna")
        if int(_metric_number(counts.get(tag))) > 0
    ]
    if present_tags:
        _line("Scenario cases:", ", ".join(present_tags))
    _line("Timing precision:", evaluation.get("timing_precision"))
    _line(
        "Global TP / FP / FN:",
        f"{summary.get('Global_TP', 0):.6f} / {summary.get('Global_FP', 0)} / {summary.get('Global_FN', 0)}",
    )
    _line(
        "Precision / Recall:",
        f"{summary.get('Precision', 0):.6f} / {summary.get('Recall', 0):.6f}",
    )

    # Hide IA-QTF1 slices that have no slots in this run (zeros are ambiguous).
    f1_slices = [
        ("1Q1A realtime:", "realtime"),
        ("1Q1A proactive:", "proactive"),
        ("1Q1A nested:", "nested"),
        ("1Q1A Global:", "one_q1a_global"),
        ("1QnA:", "one_qna"),
    ]
    present_f1 = [(label, key) for label, key in f1_slices if _slice_slots(key) > 0]
    has_all_global = int(_metric_number(summary.get("num_slots"))) > 0
    if present_f1 or has_all_global:
        print(_report_rule("-", "IA-QTF1"))
        if exp_f1:
            for label, key in present_f1:
                _f1_line(label, key)
            if has_all_global:
                _f1_line("All Global:", "all_global")
        else:
            _line("All Global:", f"{summary.get('IA_QTF1', 0):.6f}")

    interrupted_slots = int(_metric_number(interruption.get("interrupted_slot_count")))
    if interrupted_slots > 0:
        print(_report_rule("-", "Interruption"))
        _line("Interrupted slots:", interrupted_slots)
        _line("NOR:", f"{interruption.get('NOR', 0):.6f}")
        _line("PAQ:", f"{interruption.get('PAQ', 0):.6f}")
        _line("CSM-SR:", f"{interruption.get('CSM_SR', 0):.6f}")
        _line("CSM-AS:", f"{interruption.get('CSM_AS', 0):.6f}s")

    nested_pairs = int(_metric_number(exp_nested.get("num_pairs", nested.get("num_pairs", 0))))
    if nested_pairs > 0:
        missed_outer = int(_metric_number(exp_nested.get("missed_outer", nested.get("missing_q1_count", 0))))
        print(_report_rule("-", "Nested"))
        _line("Nested pairs:", nested_pairs)
        _line("NCCS:", f"{_metric_number(exp_nested.get('NCCS', nested.get('NCCS', 0))):.6f}")
        _line(
            "Inner IA-QTF1:",
            f"{_metric_number(exp_nested.get('inner_IA_QTF1', 0)):.6f}",
        )
        _line(
            "Outer IA-QTF1:",
            f"{_metric_number(exp_nested.get('outer_IA_QTF1', 0)):.6f}",
        )
        _line("Missed outer:", f"{missed_outer} / {nested_pairs}")

    print(_report_rule("="))
    print()


def evaluate_batch(
    cases: list[OmniInteractCase],
    results: list[OmniInteractCaseResult],
    options: OmniInteractEvaluationOptions,
) -> dict[str, object]:
    """Evaluate eligible cases, persist details, and return a compact summary."""

    output_root = options.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    judge = OmniInteractJudge(
        options.judge_base_url,
        options.judge_model,
        api_key=options.judge_api_key,
        timeout_s=options.judge_timeout_s,
        max_tokens=options.judge_max_tokens,
    )
    item_rows: list[dict[str, object]] = []
    work: list[tuple[OmniInteractCase, OmniInteractCaseResult, Path, EvaluationInputsFingerprint | None]] = []
    skipped = 0
    for case, result in zip(cases, results, strict=True):
        if not result.success or not result.eligible_for_official_eval:
            skipped += 1
            continue
        sample_id = f"{case.subset}__{Path(result.output_dir).name}"
        destination = output_root / f"{sample_id}.unified_eval.json"
        expected: EvaluationInputsFingerprint | None
        try:
            expected = _fingerprint_from_options(case, result, options)
        except OSError:
            expected = None
        if expected is not None and options.skip_existing and destination.is_file():
            existing = _mapping(_read_json(destination))
            cached = EvaluationInputsFingerprint.from_mapping(existing.get("inputs_fingerprint"))
            if existing.get("status") == "ok" and cached == expected:
                item_rows.append(existing)
                continue
        work.append((case, result, destination, expected))

    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        futures = {
            executor.submit(
                evaluate_case,
                case,
                result,
                judge,
                destination,
                inputs_fingerprint=fingerprint,
            ): (case, destination)
            for case, result, destination, fingerprint in work
        }
        for future in as_completed(futures):
            case, destination = futures[future]
            try:
                item_rows.append(future.result())
            except (JudgeRequestError, OSError, ValueError) as exc:
                failures.append(
                    {
                        "sample_id": destination.name.removesuffix(".unified_eval.json"),
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    slot_rows = [slot for item in item_rows for slot in _items(item.get("slots")) if isinstance(slot, dict)]
    unmatched = sum(
        int(summary.get("num_unmatched_chunks", 0))
        for item in item_rows
        if isinstance((summary := item.get("summary")), dict)
    )
    timing_precision = (
        "word-aligned"
        if item_rows and all(item.get("timing_precision") == "word-aligned" for item in item_rows)
        else "chunk-level-approximate"
    )
    evaluation = {
        "status": "ok" if not failures else "partial" if item_rows else "failed",
        "judge_model": options.judge_model,
        "judge_base_url": options.judge_base_url,
        "protocol_source": PROTOCOL_SOURCE,
        "total": len(cases),
        "evaluated": len(item_rows),
        "failed": len(failures),
        "skipped": skipped,
        "timing_precision": timing_precision,
        "summary": _summarize(slot_rows, unmatched),
        "items": [
            {
                "sample_id": item.get("sample_id"),
                "status": item.get("status"),
                "summary": item.get("summary"),
                "timing_precision": item.get("timing_precision"),
            }
            for item in item_rows
        ]
        + failures,
    }
    _write_json(output_root / "unified_eval_summary.json", evaluation)
    print_evaluation_report(evaluation)
    return evaluation
