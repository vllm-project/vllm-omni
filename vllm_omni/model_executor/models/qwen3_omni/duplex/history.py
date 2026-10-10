# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Conservative original-text prefix selection for interrupted Qwen speech.

ASR text is evidence only: callers always slice the original Thinker text.
Matching and conservative boundaries reduce uncertainty but do not establish
exact acoustic alignment.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

import regex as re


@dataclass(frozen=True)
class PrefixResult:
    """Original-text character offset, or a reason to reject the evidence."""

    char_end: int
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return not self.reason


@dataclass(frozen=True)
class TextUnit:
    """Normalized text unit and its start/inclusive, end/exclusive original span."""

    normalized: str
    start: int
    end: int


# Keep original offsets: normalizing the whole string first can change lengths.
_UNITS = re.compile(r"[\u3400-\u9fff]|[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_PROTECTED = frozenset({"不", "没", "无", "非", "别", "勿", "未", "not", "no", "never", "cannot", "don't", "can't"})


def text_units(text: str) -> list[TextUnit]:
    """Chinese characters / other words, with reversible original offsets."""
    # The second regex alternative must not absorb Chinese following Latin text.
    result = []
    for match in _UNITS.finditer(text):
        for part in re.finditer(r"[\u3400-\u9fff]|[^\u3400-\u9fff]+", match.group()):
            value = unicodedata.normalize("NFKC", part.group()).casefold().replace("’", "'")
            result.append(TextUnit(value, match.start() + part.start(), match.start() + part.end()))
    return result


def _protected(value: str) -> bool:
    return value in _PROTECTED or any(c.isdigit() for c in value)


def asr_prefix(
    original: str,
    transcript: str,
    *,
    max_error_rate: float = 0.2,
    tail_guard_units: int = 1,
    max_units: int = 512,
) -> PrefixResult:
    """Align all ASR units to an anchored original prefix, allowing edits.

    Original suffixes are free, original leading deletions are forbidden. Equal
    best endpoints are refused. Sensitive deletions/substitutions, long gaps,
    and an unanchored tail are refused too. Withhold the final recognized unit
    because ASR can complete a word cut in half even without a spelling error.
    """
    if not 0 <= max_error_rate < 1 or tail_guard_units < 1 or max_units < 1:
        raise ValueError("Invalid ASR prefix limits")
    source, heard = text_units(original), text_units(transcript)
    n, m = len(source), len(heard)
    if m < 3 or not n:
        return PrefixResult(0, "insufficient_text")
    if max(n, m) > max_units:
        return PrefixResult(0, "text_limit")
    inf = n + m + 1
    costs = [[inf] * (n + 1) for _ in range(m + 1)]
    costs[0][0] = 0
    for i in range(1, m + 1):
        costs[i][0] = i
        for j in range(1, n + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (heard[i - 1].normalized != source[j - 1].normalized),
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )
    best = min(costs[m][1:])
    if best > math.floor(m * max_error_rate):
        return PrefixResult(0, "edit_distance")
    ends = [j for j in range(1, n + 1) if costs[m][j] == best]
    if len(ends) != 1:
        return PrefixResult(0, "ambiguous_endpoint")
    i, j = m, ends[0]
    # A tied edit path is ambiguous even if its final endpoint is unique.
    matches: list[tuple[int, int]] = []
    tail_matches = 0
    in_tail = True
    deleted = 0
    while i or j:
        choices = []
        if i and j:
            equal = heard[i - 1].normalized == source[j - 1].normalized
            if costs[i][j] == costs[i - 1][j - 1] + (not equal):
                choices.append("match" if equal else "replace")
        if i and costs[i][j] == costs[i - 1][j] + 1:
            choices.append("insert")
        if j and costs[i][j] == costs[i][j - 1] + 1:
            choices.append("delete")
        if len(choices) != 1:
            return PrefixResult(0, "ambiguous_path")
        op = choices[0]
        if op == "match":
            matches.append((i - 1, j - 1))
            if in_tail:
                tail_matches += 1
            deleted = 0
        else:
            in_tail = False
            if op in {"delete", "replace"} and _protected(source[j - 1].normalized):
                return PrefixResult(0, "sensitive_edit")
            if op in {"insert", "replace"} and _protected(heard[i - 1].normalized):
                return PrefixResult(0, "sensitive_edit")
            deleted = deleted + 1 if op == "delete" else 0
            if deleted > 1:
                return PrefixResult(0, "deletion_gap")
        i -= op != "delete"
        j -= op != "insert"
    if tail_matches < tail_guard_units + 2:
        return PrefixResult(0, "unanchored_tail")
    eligible = [s for h, s in matches if h < m - tail_guard_units]
    return PrefixResult(source[max(eligible)].end if eligible else 0)
