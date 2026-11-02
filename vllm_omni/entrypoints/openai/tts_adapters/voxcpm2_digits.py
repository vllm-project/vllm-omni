# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Edge digit-run prosody shaping for the VoxCPM2 text frontend.

VoxCPM2's acoustic stream mishandles digit-runs at utterance edges under
zero-shot/anchored rendering. The text layer (TextNormalizer) is never the
culprit - it normalizes every shape correctly; the failures are the model's
acoustic handling of the prosodic context around edge digits, so the fix
shapes that context:

1. Leading lone 十 does not voice. The tokenizer lacks fused forms for
   ▁零/▁六/▁八/▁九/▁十/▁两; a leading 十 absorbed into the cold-start pause
   goes silent. A comma lead-in ("10今天..." -> "10，今天...") gives the
   digit a voiced neighbor and reads 十 naturally.
2. Trailing digits split digit-by-digit under a comma but voice under a
   sentence-final stop. Measured (n=100 rendered + ASR): a trailing "十，"
   splits to "一零" at 87% while "十。" reads 十 at 92%; the head/tail
   asymmetry is real (a comma head measured 9% garble vs a period head's
   23% with a "1"->"七" misread), so the two edges take DIFFERENT
   punctuation: comma leads, full stop closes.
3. A lone edge '0' is held out of TextNormalizer with a sentinel token:
   normalized 零 is swallowed at both edges, and the raw digit token gives a
   weak-but-right-syllable líng. The lookbehind matters: in "10。" the 0 is
   the second digit of the run - sentinel-splitting it would emit
   "1<sentinel>。" i.e. "一零" (measured: 77% of an n=100 batch), which is
   exactly the failure the run-pad exists to prevent.
4. The comma lead-in applies ONLY to standalone edge numbers. When the
   leading digit is bound to a unit/date/time/measure word (月日年号点分秒
   万元...), the digit is part of a normal numeric expression, and the
   inserted comma does not help: "3点到" garbles more WITH the comma
   (measured n=5: clean 三点到 2/5 plain vs garble-heavy with comma) while
   "1月1日开会" / "10月份见" / "2万元已到账" read perfectly without it
   (n=3 each). A following bound character marks the digit as "used up",
   not as an edge-run that needs a voiced neighbor.
"""

from __future__ import annotations

import os
from typing import Any

import regex as re

_VOXCPM2_ZERO_SENTINEL = "\uee00"
# Characters that bind a preceding digit into a normal numeric expression
# (dates 月日年号, clock times 点分秒, magnitudes 万亿 + currencies 元角,
# measure words 个只条根张片本件台位名次遍趟层楼岁 and units 斤两米尺寸
# 升斗吨磅周). The comma lead-in skips digit-runs directly followed by
# one of these - measured: the inserted comma never helps and sometimes
# garbles those readings (n=3..5 per shape), while standalone edge
# numbers ("10今天...") need it.
_VOXCPM2_BOUND_FOLLOW_RE = re.compile(
    r"[月日年号点分秒万亿两元角块个只条根张片本件台位名次遍趟层楼岁斤米尺寸升斗吨磅周]"
)
_VOXCPM2_LEAD_TEN_RE = re.compile(r"^10(?=[\u4e00-\u9fff])")
_VOXCPM2_LEADING_DIGIT_RUN_RE = re.compile(r"^(\d{1,2})(?=[\u4e00-\u9fff])")
_VOXCPM2_TRAILING_DIGIT_RUN_RE = re.compile(r"([\u4e00-\u9fff。！？，、；：])(\d{1,2})$")
_VOXCPM2_TN_TRIGGER_RE = re.compile(r"[\d$%^*+=<>√≈²³]")
_VOXCPM2_LONE_TRAILING_ZERO_RE = re.compile(r"(?<![\d.])0。$")
_LONE_LEADING_ZERO_RE = re.compile(r"^0，")


def _leading_digit_run_pad(m: re.Match[str]) -> str:
    """Pad a leading 1-2 digit run with a comma - but only when the run is a
    standalone edge number, not bound to a following unit/date/measure char
    ("1月1日开会" keeps "1月"; "3点到" keeps "3点"; "10今天..." pads)."""
    digit_run = m.group(1)
    rest = m.string[m.end() :]
    if _VOXCPM2_BOUND_FOLLOW_RE.match(rest):
        return digit_run
    return digit_run + "，"


def shape_voxcpm2_edge_digits(text: str) -> str:
    """Prosody-shape edge digit-runs (comma lead-in, sentence-final stop,
    lone-zero sentinel) before TextNormalizer sees the text."""
    if not text:
        return text
    if os.environ.get("VLLM_OMNI_SPEECH_EDGE_PAD", "1").strip().lower() in {"0", "false", "no", "off"}:
        return text
    # Leading "10" gets the same comma lead-in as single digits - unless a
    # bound char follows ("10月份见" is a date, not an edge run).
    if _VOXCPM2_LEAD_TEN_RE.match(text) and not _VOXCPM2_BOUND_FOLLOW_RE.match(text[2:]):
        text = "10，" + text[2:]
    text = _VOXCPM2_LEADING_DIGIT_RUN_RE.sub(_leading_digit_run_pad, text)
    # Trailing digit-run -> FULL STOP, never a comma (see module docstring).
    # Three entry shapes: digit + ， -> replace with 。; bare digit ->
    # append 。; digit already followed by 。 -> untouched.
    if re.fullmatch(r"\d{1,2}", text):
        # ZhIMQ's simplification (the whole text IS a 1-2 digit run)
        text = f"{text}。"
    elif text.endswith("，"):
        core = text[:-1]
        if _VOXCPM2_TRAILING_DIGIT_RUN_RE.search(core) or re.search(r"\d{1,2}$", core):
            text = core + "。"
    elif not re.search(r"[。！？]$", text):
        text = _VOXCPM2_TRAILING_DIGIT_RUN_RE.sub(r"\1\2。", text)
    # Hold a LONE edge '0' out of TN conversion. The lookbehind matters: in
    # "10。" the 0 is the second digit of the run - sentinel-splitting it
    # would emit "1<sentinel>。" i.e. "一零".
    # Slice by the MATCH SPAN, not text[:-2]: "$" also matches before a
    # final newline, so for "今天0。\n" the matched "0。" is NOT the last two
    # characters of the string - a [:-2] slice would drop "。\n" and re-append
    # the stop after the sentinel, yielding "今天00。" (an extra digit) when
    # the sentinel is restored with normalization off.
    zero_match = _VOXCPM2_LONE_TRAILING_ZERO_RE.search(text)
    if zero_match:
        text = text[: zero_match.start()] + _VOXCPM2_ZERO_SENTINEL + "。" + text[zero_match.end() :]
    return text


def normalize_voxcpm2_text(text: str, text_normalizer: Any) -> str:
    """Run TextNormalizer when the text contains TN-trigger characters, then
    restore sentinel-held digits. A lone leading "0，" is held out of TN with
    the sentinel (normalized 零 is swallowed at the leading edge; the raw
    digit token voices a weak-but-right líng).

    ``text_normalizer=None`` (the optional dependency unavailable) keeps the
    documented TN-off path: the sentinel still applies and is restored, only
    the numeric normalization is skipped."""
    if text and _LONE_LEADING_ZERO_RE.match(text):
        text = _VOXCPM2_ZERO_SENTINEL + text[1:]
    if (
        text
        and text_normalizer is not None
        and os.environ.get("VLLM_OMNI_SPEECH_TEXT_NORMALIZE", "1").strip().lower() not in {"0", "false", "no", "off"}
        and _VOXCPM2_TN_TRIGGER_RE.search(text)
    ):
        normalized = text_normalizer.normalize(text)
        if isinstance(normalized, str):
            text = normalized
    return text.replace(_VOXCPM2_ZERO_SENTINEL, "0")
