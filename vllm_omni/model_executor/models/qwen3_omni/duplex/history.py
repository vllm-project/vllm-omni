# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Conservative original-text prefix selection for interrupted Qwen speech.

ASR text is evidence only: callers always slice the original Thinker text.
Matching and conservative boundaries reduce uncertainty but do not establish
exact acoustic alignment.
"""

from __future__ import annotations

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


def text_units(text: str) -> list[TextUnit]:
    """Chinese characters / other words, with reversible original offsets."""
    # The second regex alternative must not absorb Chinese following Latin text.
    result = []
    for match in _UNITS.finditer(text):
        for part in re.finditer(r"[\u3400-\u9fff]|[^\u3400-\u9fff]+", match.group()):
            value = unicodedata.normalize("NFKC", part.group()).casefold().replace("’", "'")
            result.append(TextUnit(value, match.start() + part.start(), match.start() + part.end()))
    return result


def asr_prefix(
    original: str,
    transcript: str,
    *,
    tail_guard_units: int = 1,
    max_units: int = 512,
) -> PrefixResult:
    """Keep the normalized common prefix, withholding its final matched units.

    Stop at the first difference: never skip, substitute or resume matching
    later text. Drop the tail of the matched prefix because ASR may complete a
    partially heard word. Valid evidence can yield an empty prefix.
    """
    if tail_guard_units < 1 or max_units < 1:
        raise ValueError("Invalid ASR prefix limits")
    source, heard = text_units(original), text_units(transcript)
    if len(heard) < 3 or not source:
        return PrefixResult(0, "insufficient_text")
    if max(len(source), len(heard)) > max_units:
        return PrefixResult(0, "text_limit")
    matched = 0
    for left, right in zip(source, heard):
        if left.normalized != right.normalized:
            break
        matched += 1
    retained = matched - tail_guard_units
    return PrefixResult(source[retained - 1].end if retained > 0 else 0)
