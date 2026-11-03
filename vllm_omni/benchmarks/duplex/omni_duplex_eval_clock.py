# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Media-clock sentence and chunk extraction and response JSON normalization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import regex as re

_TERMINAL = re.compile(r"(.+?[.!?。！？]+)(?:\s+|$)", re.S)
_TEXT_DELTA_TYPES = frozenset(
    {
        "response.output_text.delta",
        "response.output_audio_transcript.delta",
        "response.text.delta",
    }
)


@dataclass
class TimedSentence:
    sentence: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {"sentence": self.sentence, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class TimedChunk:
    text: str
    current_time: float

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "current_time": round(self.current_time, 3)}


def split_text(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts: list[str] = []
    position = 0
    for match in _TERMINAL.finditer(text):
        parts.append(match.group(1).strip())
        position = match.end()
    tail = text[position:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def _text_deltas(events: Iterable[dict[str, Any]]) -> Iterator[tuple[str, float | None]]:
    """Yield output text deltas with their input media time; whitespace-only deltas have none."""
    last: float | None = None
    for event in events:
        if event.get("type") not in _TEXT_DELTA_TYPES:
            continue
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            continue
        if not delta.strip():
            yield delta, None
            continue
        timestamp = event.get("audio_end_ms")
        if timestamp is None:
            timestamp = event.get("_media_clock_ms", event.get("current_time_ms", 0.0))
        try:
            # Every accepted event field is explicitly millisecond-valued.
            now = float(timestamp) / 1000.0
        except (TypeError, ValueError):
            now = last or 0.0
        last = now
        yield delta, now


def extract_timed_sentences(events: Iterable[dict[str, Any]], *, clock: str = "media") -> list[TimedSentence]:
    """Split output deltas while anchoring text to input ``audio_end_ms``."""
    if clock != "media":
        raise ValueError("only clock=media is supported")
    sentences: list[TimedSentence] = []
    pending = ""
    start: float | None = None
    last: float | None = None
    for delta, now in _text_deltas(events):
        if now is None:
            # Keep the space between tokens (Qwen emits " " before a digit on
            # its own), but never open a sentence on whitespace.
            if pending:
                pending += delta
            continue
        pending += delta
        start = now if start is None else start
        last = now
        pieces = split_text(pending)
        complete = pieces if pieces and re.search(r"[.!?。！？]$", pending.strip()) else pieces[:-1]
        if complete:
            for piece in complete:
                sentences.append(TimedSentence(piece, start, max(start, now)))
            consumed = " ".join(complete)
            pending = pending[len(consumed) :].strip()
            start = now if pending else None
    if pending.strip():
        sentences.append(TimedSentence(pending.strip(), start or last or 0.0, last or start or 0.0))
    return sentences


def extract_timed_chunks(events: Iterable[dict[str, Any]], *, clock: str = "media") -> list[TimedChunk]:
    """Group consecutive output deltas that share one input media time.

    Each group is one model output chunk. Proactive-reminder scoring places
    chunks, not whole sentences, in the reminder window, so a sentence that
    starts before the window still contributes the text emitted inside it.
    """
    if clock != "media":
        raise ValueError("only clock=media is supported")
    chunks: list[TimedChunk] = []
    text = ""
    current: float | None = None
    for delta, now in _text_deltas(events):
        if now is None:
            if text:
                text += delta
            continue
        if current is not None and now != current:
            if text.strip():
                chunks.append(TimedChunk(text, current))
            text = ""
        text += delta
        current = now
    if current is not None and text.strip():
        chunks.append(TimedChunk(text, current))
    return chunks


def has_response_chunks(raw: Any) -> bool:
    """Whether a response JSON records per-chunk output times."""
    return isinstance(raw, dict) and isinstance(raw.get("chunks"), list)


def normalize_response_chunks(raw: Any) -> list[dict[str, Any]]:
    """Read timed chunks for proactive-reminder scoring.

    Published responses carry ``chunks``; a bare sentence list is scored by
    sentence start, as before chunks were recorded.
    """
    if has_response_chunks(raw):
        items = raw["chunks"]
    elif isinstance(raw, dict) and isinstance(raw.get("sentences"), list):
        items = raw["sentences"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("response JSON must be a list or contain chunks/sentences")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text", item.get("sentence", item.get("content", "")))
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            current_time = float(item.get("current_time", item.get("start", item.get("time", 0.0))) or 0.0)
        except (TypeError, ValueError):
            continue
        # Keep the chunk's own whitespace. Scoring concatenates chunks with no
        # extra separator, so stripping here would split a word or insert a
        # space into CJK when the pieces are joined.
        normalized.append({"text": text, "current_time": current_time})
    return normalized


def normalize_response_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        items = raw.get("sentences", raw.get("chunks"))
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError("response JSON must be a list or contain sentences/chunks")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("sentence", item.get("text", item.get("content", "")))
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            start = float(item.get("start", item.get("current_time", item.get("time", 0.0))) or 0.0)
            end = float(item.get("end", item.get("current_time", item.get("time", start))) or start)
        except (TypeError, ValueError):
            continue
        normalized.append({"sentence": text.strip(), "start": start, "end": max(start, end)})
    return normalized


def validate_clock(meta: dict[str, Any], *, allow_invalid: bool = False) -> None:
    if meta.get("clock") == "invalid" and not allow_invalid:
        raise ValueError("generate artifact has clock=invalid; pass --allow-invalid-clock to score it")
