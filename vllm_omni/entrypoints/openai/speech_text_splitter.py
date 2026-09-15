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

# ASCII punctuation is ambiguous (abbreviations, decimals, thousands
# separators), so it only closes a unit when whitespace or a flush follows.
# Script-specific marks below are unambiguous and close a unit as soon as the
# character after them is known.
_ASCII_SENTENCE_TERMINATORS = frozenset(".!?")
_ASCII_CLAUSE_TERMINATORS = _ASCII_SENTENCE_TERMINATORS | frozenset(",;")
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
    "，",
    "；",
    "،",  # Arabic comma
    "؛",  # Arabic semicolon
}

_SENTENCE_TERMINATORS = _ASCII_SENTENCE_TERMINATORS | _SCRIPT_SENTENCE_TERMINATORS
_CLAUSE_TERMINATORS = _ASCII_CLAUSE_TERMINATORS | _SCRIPT_CLAUSE_TERMINATORS

# Trailing characters that belong to the unit they close, so `Wait...` and
# `He said "Hello."` stay one TTS request instead of several.
_CLOSING_DELIMITERS = frozenset("\"'”’»)]}』」）】")
_RUN_CHARS = _CLAUSE_TERMINATORS | _CLOSING_DELIMITERS

# Abbreviations that end in `.` mid-sentence. Single letters are covered
# separately so initials such as `J. R. R.` do not split either.
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "mt",
        "fig",
        "no",
        "vs",
        "etc",
        "approx",
        "dept",
        "inc",
        "ltd",
        "co",
        "al",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
        "a.m",
        "p.m",
    }
)
_MAX_ABBREVIATION_LEN = max(len(abbrev) for abbrev in _ABBREVIATIONS)


def _is_token_char(ch: str) -> bool:
    return ch.isalnum() or ch == "."


def _is_numeric_separator(buffer: str, index: int) -> bool:
    """True for the `.`/`,` inside `3.14` or `1,000`."""
    if buffer[index] not in ".,":
        return False
    if index == 0 or not buffer[index - 1].isdigit():
        return False
    return index + 1 < len(buffer) and buffer[index + 1].isdigit()


def _is_abbreviation(buffer: str, index: int) -> bool:
    """True for the `.` of `Dr.`, `e.g.`, or an initial like `J.`."""
    if buffer[index] != ".":
        return False
    # Bounded so a frame of `a.a.a...` cannot walk the whole prefix at every
    # `.`. Every abbreviation fits inside the limit, so hitting it can only
    # rule a token out, never produce a false positive.
    limit = max(0, index - _MAX_ABBREVIATION_LEN)
    start = index
    while start > limit and _is_token_char(buffer[start - 1]):
        start -= 1
    if start > 0 and _is_token_char(buffer[start - 1]):
        return False
    token = buffer[start:index]
    if not token:
        return False
    if len(token) == 1 and token.isalpha():
        return True
    return token.lower().strip(".") in _ABBREVIATIONS


def extract_complete_units(
    buffer: str,
    terminators: frozenset[str],
    *,
    flush: bool,
    scan_from: int = 0,
) -> tuple[list[str], str, int]:
    """Split ``buffer`` into complete units.

    Returns ``(units, remainder, next_scan)``. ``next_scan`` is an offset into
    ``remainder``: everything before it has been examined already, so a caller
    appending more text can pass it back as ``scan_from`` instead of rescanning
    the whole buffer on every ``input.text`` message.
    """
    units: list[str] = []
    last_split = 0
    i = max(scan_from, 0)
    length = len(buffer)
    while i < length:
        ch = buffer[i]
        if ch not in terminators or _is_numeric_separator(buffer, i) or _is_abbreviation(buffer, i):
            i += 1
            continue

        # Absorb the punctuation/closing-delimiter run so `Wait...` emits once.
        run_end = i + 1
        while run_end < length and buffer[run_end] in _RUN_CHARS:
            run_end += 1

        if run_end >= length and not flush:
            # The run may still grow; re-examine this terminator next feed.
            return units, buffer[last_split:], i - last_split

        ascii_terminator = ch in _ASCII_CLAUSE_TERMINATORS
        if run_end < length and ascii_terminator and not buffer[run_end].isspace():
            i = run_end
            continue

        piece = buffer[last_split:run_end].strip()
        if piece:
            units.append(piece)
        i = run_end
        while i < length and buffer[i].isspace():
            i += 1
        last_split = i

    remainder = buffer[last_split:]
    if flush:
        tail = remainder.strip()
        if tail:
            units.append(tail)
        return units, "", 0
    return units, remainder, len(remainder)


class SpeechTextSplitter:
    """Stateful splitter used by one WebSocket utterance."""

    def __init__(self, granularity: str = "none") -> None:
        if granularity not in ("none", "sentence", "clause"):
            raise ValueError(f"Unsupported split_granularity: {granularity!r}")
        self.granularity = granularity
        self._buf = ""
        self._scan = 0

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
        units, self._buf, self._scan = extract_complete_units(self._buf, terminators, flush=False, scan_from=self._scan)
        return units

    def flush(self) -> list[str]:
        terminators = self._terminators()
        if terminators is None:
            piece = self._buf.strip()
            self._buf = ""
            self._scan = 0
            return [piece] if piece else []
        units, self._buf, self._scan = extract_complete_units(self._buf, terminators, flush=True, scan_from=self._scan)
        return units
