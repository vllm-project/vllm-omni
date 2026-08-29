# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental text splitting for WebSocket TTS input.

Default serving still synthesizes one request per ``input.done`` (see
``split_granularity='none'``). Opting into ``sentence`` or ``clause`` restores
per-boundary requests so incremental STT/LLM clients can start audio before
the full utterance arrives.

Terminators cover Latin, CJK, Indic danda, and Arabic question/semicolon marks
so Hindi and similar scripts are not stuck waiting for ``.`` / ``?``.
"""

from __future__ import annotations

# ASCII `.!?` are abbreviation-prone; they need following whitespace (or a
# flush) before we treat them as complete. Script-specific marks below are
# almost never abbreviations, so they close a unit immediately.
_LATIN_TERMINATORS = frozenset(".!?")
_SCRIPT_SENTENCE_TERMINATORS = frozenset(
    {
        "。",
        "！",
        "？",
        "…",
        "।",  # Devanagari danda
        "॥",  # Devanagari double danda
        "؟",  # Arabic question mark
    }
)
_SCRIPT_CLAUSE_TERMINATORS = _SCRIPT_SENTENCE_TERMINATORS | {
    ",",
    ";",
    "，",
    "；",
    "،",  # Arabic comma
}

_SENTENCE_TERMINATORS = _LATIN_TERMINATORS | _SCRIPT_SENTENCE_TERMINATORS
_CLAUSE_TERMINATORS = _LATIN_TERMINATORS | _SCRIPT_CLAUSE_TERMINATORS


def _is_decimal_point(buffer: str, index: int) -> bool:
    if buffer[index] != ".":
        return False
    if index == 0 or not buffer[index - 1].isdigit():
        return False
    return index + 1 < len(buffer) and buffer[index + 1].isdigit()


def extract_complete_units(buffer: str, terminators: frozenset[str], *, flush: bool) -> tuple[list[str], str]:
    """Split ``buffer`` into complete units; return (units, remainder)."""
    units: list[str] = []
    last_split = 0
    i = 0
    length = len(buffer)
    while i < length:
        ch = buffer[i]
        if ch not in terminators or _is_decimal_point(buffer, i):
            i += 1
            continue

        j = i + 1
        while j < length and buffer[j].isspace():
            j += 1

        latin = ch in _LATIN_TERMINATORS
        if latin:
            complete = j > i + 1 or flush
        else:
            complete = True

        if not complete:
            i += 1
            continue

        piece = buffer[last_split:j].strip()
        if piece:
            units.append(piece)
        last_split = j
        i = j

    remainder = buffer[last_split:]
    if flush:
        tail = remainder.strip()
        remainder = ""
        if tail:
            units.append(tail)
    return units, remainder


class SpeechTextSplitter:
    """Stateful splitter used by one WebSocket utterance."""

    def __init__(self, granularity: str = "none") -> None:
        if granularity not in ("none", "sentence", "clause"):
            raise ValueError(f"Unsupported split_granularity: {granularity!r}")
        self.granularity = granularity
        self._buf = ""

    def has_buffered_text(self) -> bool:
        return bool(self._buf)

    def _terminators(self) -> frozenset[str] | None:
        if self.granularity == "none":
            return None
        if self.granularity == "clause":
            return _CLAUSE_TERMINATORS
        return _SENTENCE_TERMINATORS

    def feed(self, text: str) -> list[str]:
        self._buf += text
        terminators = self._terminators()
        if terminators is None:
            return []
        units, self._buf = extract_complete_units(self._buf, terminators, flush=False)
        return units

    def flush(self) -> list[str]:
        terminators = self._terminators()
        if terminators is None:
            piece = self._buf.strip()
            self._buf = ""
            return [piece] if piece else []
        units, self._buf = extract_complete_units(self._buf, terminators, flush=True)
        return units
