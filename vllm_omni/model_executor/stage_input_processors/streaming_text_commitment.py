# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packet-boundary-independent commitment policy for streaming TTS text.

The policy owns only source-text commitment. It identifies complete lexical
and special-text atoms, but deliberately leaves model-specific normalization to
the consumer. Transport packet boundaries never close an otherwise open atom.

The deterministic grammar is intentionally small. It covers the initial RFC
scope of Chinese and English numbers, common units and symbols, and unfinished
ASCII words or abbreviations. Natural-language text outside that grammar is
committed immediately; this module does not claim general polyphone safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import regex as re

SpanKind = Literal["natural", "lexical", "special"]

_FULLWIDTH_ALNUM_RE = re.compile(r"[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]")
_DIGIT_RE = re.compile(r"[0-9０-９]")

_LEADING_SPECIAL_SYMBOLS = frozenset("$¥￥€£₽+-−—±~≈<>≤≥=×÷*/\\|@#^&_%％°℃℉")
_BODY_SYMBOLS = frozenset("$¥￥€£₽+-−—±~≈<>≤≥=×÷*/\\|@#^&_%％°℃℉²³()[]{}（）【】_'\"")
_AMBIGUOUS_PUNCTUATION = frozenset(".,:．，：")
_LEADING_DECIMAL_POINTS = frozenset(".．")
_STRONG_SENTENCE_END = frozenset(".!?。！？…\n")
_PROFILE = "zh_en_special_v1"
_KEYCAP_MARK = "\N{COMBINING ENCLOSING KEYCAP}"
_KEYCAP_VARIATION = "\N{VARIATION SELECTOR-16}"
_DOTTED_ABBREVIATION_RE = re.compile(r"(?:[A-Za-z]\.){2,}\Z")

# Units are tokens, not a bag of characters. Prefix-aware matching is needed
# for packet seams such as ``3千瓦|时`` and ``3米|²``: the shorter unit must not
# be committed while the buffered suffix can still become a longer known unit.
_CJK_UNITS = frozenset(
    {
        "年",
        "月",
        "日",
        "日元",
        "号",
        "时",
        "小时",
        "点",
        "分",
        "分钟",
        "秒",
        "周",
        "季",
        "度",
        "摄氏度",
        "华氏度",
        "元",
        "角",
        "圆",
        "美元",
        "欧元",
        "人民币",
        "米",
        "米²",
        "米³",
        "公里",
        "千米",
        "厘米",
        "厘米²",
        "厘米³",
        "毫米",
        "微米",
        "纳米",
        "平方米",
        "平方公里",
        "立方米",
        "立方厘米",
        "克",
        "公斤",
        "千克",
        "斤",
        "吨",
        "升",
        "毫升",
        "伏",
        "安",
        "瓦",
        "千瓦",
        "千瓦时",
        "赫兹",
        "兆赫兹",
        "吉赫兹",
        "成",
        "折",
        "倍",
    }
)

_ASCII_UNITS = frozenset(
    {
        "amu",
        "c",
        "cm",
        "cm2",
        "cm3",
        "cm²",
        "cm³",
        "db",
        "dm",
        "dm2",
        "dm3",
        "dm²",
        "dm³",
        "f",
        "ft",
        "g",
        "gb",
        "ghz",
        "gw",
        "gwh",
        "h",
        "ha",
        "hz",
        "kb",
        "kg",
        "khz",
        "km",
        "km2",
        "km3",
        "km²",
        "km³",
        "l",
        "lb",
        "m",
        "m2",
        "m3",
        "m²",
        "m³",
        "mb",
        "mhz",
        "min",
        "ml",
        "mm",
        "ms",
        "mw",
        "nm",
        "nd",
        "oz",
        "rd",
        "s",
        "st",
        "tb",
        "th",
        "v",
        "w",
        "°c",
        "°f",
    }
)

_ALL_UNITS = _CJK_UNITS | _ASCII_UNITS


class CommitmentState(str, Enum):
    """Terminal lifecycle for one logical streaming-text request."""

    OPEN = "open"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class CommittedTextSpan:
    """One irreversible raw-source span released by the policy."""

    source_text: str
    kind: SpanKind
    boundary_after: bool = False

    @property
    def text(self) -> str:
        """Raw source alias for consumers that do not need provenance naming."""

        return self.source_text


@dataclass(frozen=True)
class CommitmentUpdate:
    """New raw spans plus the source suffix still withheld by the policy."""

    spans: tuple[CommittedTextSpan, ...]
    pending_text: str
    final: bool = False

    @property
    def committed_text(self) -> str:
        return "".join(span.source_text for span in self.spans)


@dataclass(frozen=True)
class _ScanResult:
    end: int
    stable_at_frontier: bool


def _is_alnum(ch: str) -> bool:
    return bool(ch and _FULLWIDTH_ALNUM_RE.fullmatch(ch))


def _is_digit(ch: str) -> bool:
    return bool(ch and _DIGIT_RE.fullmatch(ch))


def _is_ascii_letter(ch: str) -> bool:
    return bool(ch and ch.isascii() and ch.isalpha())


def _is_lexical_start(ch: str) -> bool:
    return _is_alnum(ch) and not _is_digit(ch)


def _is_special_start(ch: str) -> bool:
    return _is_digit(ch) or ch in _LEADING_SPECIAL_SYMBOLS


def _is_special_start_at(text: str, index: int) -> bool:
    """Return whether ``index`` starts a recognized special-text atom.

    A decimal point followed by a digit begins a leading decimal such as
    ``.5``.  A point at the transport frontier is retained by the natural-text
    scanner until following input or EOF disambiguates it.
    """

    ch = text[index]
    if _is_special_start(ch):
        return True
    if ch not in _LEADING_DECIMAL_POINTS:
        return False
    if ch == "." and index > 0 and text[index - 1] == ".":
        # In ``Wait...5``, the last point belongs to the repeated-dot run; it
        # must not be reclassified as the start of ``.5``. Other sentence
        # boundaries, such as a newline or ``。``, can precede a new decimal.
        return False
    following = text[index + 1] if index + 1 < len(text) else ""
    return _is_digit(following)


def _unit_prefix_state(value: str) -> tuple[bool, bool]:
    """Return ``(is_complete, can_extend)`` for a case-insensitive unit."""

    folded = value.casefold()
    is_complete = any(unit.casefold() == folded for unit in _ALL_UNITS)
    can_extend = any(unit.casefold().startswith(folded) and unit.casefold() != folded for unit in _ALL_UNITS)
    return is_complete, can_extend


def _match_unit(text: str, start: int) -> tuple[int | None, int, bool]:
    """Match a unit token with longest-complete backtracking.

    Returns the longest complete end, the end of the recognized prefix, and
    whether that prefix reaches the current transport frontier.
    """

    index = start
    last_complete: int | None = None
    candidate = ""
    while index < len(text):
        next_candidate = candidate + text[index]
        complete, can_extend = _unit_prefix_state(next_candidate)
        if not complete and not can_extend:
            break
        candidate = next_candidate
        index += 1
        if complete:
            last_complete = index
        if not can_extend:
            break
    return last_complete, index, index == len(text) and bool(candidate)


def _next_nonspace(text: str, start: int) -> tuple[int, str]:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    return index, text[index] if index < len(text) else ""


def _scan_lexical(text: str, start: int, *, final: bool) -> _ScanResult:
    index = start
    while index < len(text):
        ch = text[index]
        if _is_alnum(ch) or ch in "²³":
            index += 1
            continue
        if ch in ".@'_+-":
            following = text[index + 1] if index + 1 < len(text) else ""
            if following and (_is_alnum(following) or following in "²³"):
                index += 1
                continue
            if ch == "." and _DOTTED_ABBREVIATION_RE.fullmatch(text[start : index + 1]):
                # The terminal dot in ``e.g.`` or ``U.S.`` belongs to the
                # lexical atom. It is not a sentence boundary even when the
                # following character is whitespace or natural-language text.
                index += 1
                continue
            if not following and not final:
                return _ScanResult(len(text), False)
        break
    return _ScanResult(index, index < len(text))


def _scan_special(text: str, start: int, *, final: bool) -> _ScanResult:
    index = start
    seen_digit = False
    while index < len(text):
        ch = text[index]
        if _is_digit(ch):
            seen_digit = True
            index += 1
            continue
        if _is_ascii_letter(ch) or ch in "²³":
            index += 1
            continue
        if ch == _KEYCAP_VARIATION:
            # VS16 and the enclosing-keycap mark are allowed to arrive in
            # separate transport packets.  Hold the whole atom while VS16 is
            # still at the frontier, then consume it with the digit whether
            # or not the optional keycap mark follows.
            if index + 1 == len(text) and not final:
                return _ScanResult(len(text), False)
            index += 1
            if text[index : index + 1] == _KEYCAP_MARK:
                index += 1
            continue
        if ch == _KEYCAP_MARK:
            index += 1
            continue
        if ch in _BODY_SYMBOLS:
            index += 1
            continue
        if ch in _AMBIGUOUS_PUNCTUATION:
            following = text[index + 1] if index + 1 < len(text) else ""
            if following and _is_alnum(following):
                index += 1
                continue
            if not following and not final:
                return _ScanResult(len(text), False)
            break
        if ch.isspace() and seen_digit:
            unit_start, following = _next_nonspace(text, index)
            if "\n" in text[index:unit_start]:
                # Newline is a documented strong boundary, never spacing
                # inside a numeric-unit atom. Leave it to the natural scanner
                # even when the next line begins with a valid unit prefix.
                break
            if not following:
                if final:
                    break
                return _ScanResult(len(text), False)
            complete_end, prefix_end, prefix_at_frontier = _match_unit(text, unit_start)
            if complete_end is not None:
                index = complete_end
                continue
            if prefix_at_frontier and not final:
                return _ScanResult(prefix_end, False)
            if following in "%％℃℉°":
                index = unit_start
                continue
            break
        if seen_digit:
            complete_end, prefix_end, prefix_at_frontier = _match_unit(text, index)
            if complete_end is not None:
                index = complete_end
                continue
            if prefix_at_frontier and not final:
                return _ScanResult(prefix_end, False)
        break

    # A transport frontier is never a linguistic boundary. Even a complete
    # unit may acquire an operator, a denominator, or another expression in the
    # next packet, so only an explicit following boundary or EOF closes it.
    return _ScanResult(index, index < len(text))


def _append_natural_spans(
    spans: list[CommittedTextSpan],
    source: str,
    *,
    hold_trailing_terminators: bool,
) -> str:
    """Append natural source and return an unresolved terminator suffix.

    Consecutive strong terminators are one boundary-bearing run.  A run at the
    transport frontier stays pending because a later packet can extend ``.``
    to ``...`` or ``?`` to ``?!``.  This avoids irreversible punctuation-only
    segments without altering the raw source.
    """

    start = 0
    index = 0
    while index < len(source):
        if source[index] not in _STRONG_SENTENCE_END:
            index += 1
            continue

        run_start = index
        while index < len(source) and source[index] in _STRONG_SENTENCE_END:
            index += 1
        if hold_trailing_terminators and index == len(source):
            if start < run_start:
                spans.append(CommittedTextSpan(source[start:run_start], "natural"))
            return source[run_start:]

        spans.append(CommittedTextSpan(source[start:index], "natural", boundary_after=True))
        start = index

    if hold_trailing_terminators and source[-1:] in _LEADING_DECIMAL_POINTS:
        if start < len(source) - 1:
            spans.append(CommittedTextSpan(source[start:-1], "natural"))
        return source[-1:]
    if start < len(source):
        spans.append(CommittedTextSpan(source[start:], "natural"))
    return ""


class _ScanState(str, Enum):
    SCAN = "scan"
    NATURAL = "natural"
    LEXICAL = "lexical"
    SPECIAL = "special"
    READY = "ready"
    HOLD = "hold"
    DONE = "done"


class _ScanEvent(str, Enum):
    NATURAL = "natural"
    LEXICAL = "lexical"
    SPECIAL = "special"
    READY = "ready"
    INCOMPLETE = "incomplete"
    ADVANCE = "advance"
    END = "end"


# Control-plane transitions are independent of the recognizers below. Adding
# a recognizer requires an explicit SCAN transition and a readiness decision;
# recognizers never emit source or choose synthesis boundaries themselves.
_SCAN_TRANSITIONS = {
    (_ScanState.SCAN, _ScanEvent.NATURAL): _ScanState.NATURAL,
    (_ScanState.SCAN, _ScanEvent.LEXICAL): _ScanState.LEXICAL,
    (_ScanState.SCAN, _ScanEvent.SPECIAL): _ScanState.SPECIAL,
    (_ScanState.SCAN, _ScanEvent.END): _ScanState.DONE,
    (_ScanState.NATURAL, _ScanEvent.ADVANCE): _ScanState.SCAN,
    (_ScanState.NATURAL, _ScanEvent.INCOMPLETE): _ScanState.HOLD,
    (_ScanState.LEXICAL, _ScanEvent.READY): _ScanState.READY,
    (_ScanState.LEXICAL, _ScanEvent.INCOMPLETE): _ScanState.HOLD,
    (_ScanState.SPECIAL, _ScanEvent.READY): _ScanState.READY,
    (_ScanState.SPECIAL, _ScanEvent.INCOMPLETE): _ScanState.HOLD,
    (_ScanState.READY, _ScanEvent.ADVANCE): _ScanState.SCAN,
}


def _source_event(text: str, index: int) -> _ScanEvent:
    """Ordered atom-start guards; a leading decimal takes special precedence."""
    if index == len(text):
        return _ScanEvent.END
    if _is_lexical_start(text[index]):
        return _ScanEvent.LEXICAL
    if _is_special_start_at(text, index):
        return _ScanEvent.SPECIAL
    return _ScanEvent.NATURAL


class _CommitmentScanner:
    """Finite-state source scanner for one transactional feed.

    SCAN classifies the next atom, a recognizer decides READY or HOLD, and
    READY is the sole emission point for lexical/special spans. Natural runs
    retain their separate maximal-terminator boundary rule. HOLD retains the
    original suffix; the next feed replays it with new input. EOF makes an
    otherwise incomplete atom READY. No normalization occurs in this machine.
    """

    def __init__(self, text: str, *, final: bool) -> None:
        self.text = text
        self.final = final
        self.cursor = 0
        self.state = _ScanState.SCAN
        self.spans: list[CommittedTextSpan] = []
        self.pending = ""
        self.candidate: CommittedTextSpan | None = None
        self.candidate_end = 0

    def _classify(self) -> _ScanEvent:
        return _source_event(self.text, self.cursor)

    def _natural(self) -> _ScanEvent:
        end = self.cursor + 1
        while _source_event(self.text, end) is _ScanEvent.NATURAL:
            end += 1
        self.pending = _append_natural_spans(
            self.spans,
            self.text[self.cursor : end],
            hold_trailing_terminators=end == len(self.text) and not self.final,
        )
        self.cursor = end
        return _ScanEvent.INCOMPLETE if self.pending else _ScanEvent.ADVANCE

    def _atom(self) -> _ScanEvent:
        kinds: dict[_ScanState, SpanKind] = {
            _ScanState.LEXICAL: "lexical",
            _ScanState.SPECIAL: "special",
        }
        recognizer = {
            _ScanState.LEXICAL: _scan_lexical,
            _ScanState.SPECIAL: _scan_special,
        }[self.state]
        result = recognizer(self.text, self.cursor, final=self.final)
        if result.end <= self.cursor:
            raise AssertionError("commitment scanner made no progress")
        if result.end == len(self.text) and not self.final and not result.stable_at_frontier:
            self.pending = self.text[self.cursor :]
            return _ScanEvent.INCOMPLETE
        self.candidate = CommittedTextSpan(self.text[self.cursor : result.end], kinds[self.state])
        self.candidate_end = result.end
        return _ScanEvent.READY

    def _emit(self) -> _ScanEvent:
        assert self.candidate is not None
        self.spans.append(self.candidate)
        self.cursor = self.candidate_end
        self.candidate = None
        return _ScanEvent.ADVANCE

    def run(self) -> tuple[tuple[CommittedTextSpan, ...], str]:
        actions = {
            _ScanState.SCAN: self._classify,
            _ScanState.NATURAL: self._natural,
            _ScanState.LEXICAL: self._atom,
            _ScanState.SPECIAL: self._atom,
            _ScanState.READY: self._emit,
        }
        while self.state not in {_ScanState.HOLD, _ScanState.DONE}:
            event = actions[self.state]()
            self.state = _SCAN_TRANSITIONS[self.state, event]
        return tuple(self.spans), self.pending


def _parse(text: str, *, final: bool) -> tuple[tuple[CommittedTextSpan, ...], str]:
    return _CommitmentScanner(text, final=final).run()


class StreamingTextCommitmentPolicy:
    """Incrementally release raw text under the scoped deterministic grammar."""

    def __init__(
        self,
        *,
        profile: str = _PROFILE,
        max_pending_chars: int = 4096,
    ) -> None:
        if profile != _PROFILE:
            raise ValueError(f"unsupported streaming text commitment profile: {profile!r}")
        if max_pending_chars <= 0:
            raise ValueError("max_pending_chars must be positive")
        self._profile = profile
        self._max_pending_chars = int(max_pending_chars)
        self._pending = ""
        self._state = CommitmentState.OPEN

    @property
    def pending_text(self) -> str:
        return self._pending

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def state(self) -> CommitmentState:
        return self._state

    @property
    def finished(self) -> bool:
        return self._state is CommitmentState.FINISHED

    def feed(self, text: str, *, final: bool = False) -> CommitmentUpdate:
        """Consume a transport chunk and return newly irreversible raw text."""

        if self._state is not CommitmentState.OPEN:
            raise RuntimeError(f"streaming text commitment policy is {self._state.value}")
        if not isinstance(text, str):
            raise TypeError("streaming text chunks must be strings")

        combined = self._pending + text
        try:
            spans, pending = _parse(combined, final=final)
            if len(pending) > self._max_pending_chars:
                raise ValueError(
                    f"unresolved streaming text suffix exceeds max_pending_chars={self._max_pending_chars}"
                )
            if final and pending:
                raise AssertionError("final commitment update retained pending text")
        except Exception:
            self._state = CommitmentState.FAILED
            raise

        # Commit request-local state only after parsing and validation succeed.
        self._pending = pending
        if final:
            self._state = CommitmentState.FINISHED
        return CommitmentUpdate(spans, pending, final=final)

    def finish(self) -> CommitmentUpdate:
        """Close the stream and release all remaining source text exactly once."""

        return self.feed("", final=True)


__all__ = (
    "CommittedTextSpan",
    "CommitmentState",
    "CommitmentUpdate",
    "StreamingTextCommitmentPolicy",
)
