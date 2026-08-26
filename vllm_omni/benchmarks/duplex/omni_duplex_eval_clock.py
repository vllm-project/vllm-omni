# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Media-clock sentence extraction and response JSON normalization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import regex as re

_TERMINAL = re.compile(r"(.+?[.!?。！？]+)(?:\s+|$)", re.S)


@dataclass
class TimedSentence:
    sentence: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {"sentence": self.sentence, "start": round(self.start, 3), "end": round(self.end, 3)}


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


def extract_timed_sentences(events: Iterable[dict[str, Any]], *, clock: str = "media") -> list[TimedSentence]:
    """Split output deltas while anchoring text to input ``audio_end_ms``."""
    if clock != "media":
        raise ValueError("only clock=media is supported")
    sentences: list[TimedSentence] = []
    pending = ""
    start: float | None = None
    last: float | None = None
    for event in events:
        event_type = event.get("type")
        if event_type not in {"response.output_text.delta", "response.audio_transcript.delta", "response.text.delta"}:
            continue
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta.strip():
            continue
        timestamp = event.get("audio_end_ms")
        if timestamp is None:
            timestamp = event.get("_media_clock_ms", event.get("current_time_ms", 0.0))
        try:
            # Every accepted event field is explicitly millisecond-valued.
            now = float(timestamp) / 1000.0
        except (TypeError, ValueError):
            now = last or 0.0
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
