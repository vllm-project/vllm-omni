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
"""

from __future__ import annotations

import os
from typing import Any

import regex as re

_VOXCPM2_ZERO_SENTINEL = "\uee00"
_VOXCPM2_LEAD_TEN_RE = re.compile(r"^10(?=[\u4e00-\u9fff])")
_VOXCPM2_LEADING_DIGIT_RUN_RE = re.compile(r"^(\d{1,2})(?=[\u4e00-\u9fff])")
_VOXCPM2_TRAILING_DIGIT_RUN_RE = re.compile(r"([\u4e00-\u9fff。！？，、；：])(\d{1,2})$")
_VOXCPM2_TN_TRIGGER_RE = re.compile(r"[\d$%^*+=<>√≈²³]")
_VOXCPM2_LONE_TRAILING_ZERO_RE = re.compile(r"(?<![\d.])0。$")
_LONE_LEADING_ZERO_RE = re.compile(r"^0，")


def shape_voxcpm2_edge_digits(text: str) -> str:
    """Prosody-shape edge digit-runs (comma lead-in, sentence-final stop,
    lone-zero sentinel) before TextNormalizer sees the text."""
    if not text:
        return text
    if os.environ.get("VLLM_OMNI_SPEECH_EDGE_PAD", "1") == "0":
        return text
    # Leading "10" gets the same comma lead-in as single digits.
    text = _VOXCPM2_LEAD_TEN_RE.sub("10，", text)
    text = _VOXCPM2_LEADING_DIGIT_RUN_RE.sub(r"\1，", text)
    # Trailing digit-run -> FULL STOP, never a comma (see module docstring).
    # Three entry shapes: digit + ， -> replace with 。; bare digit ->
    # append 。; digit already followed by 。 -> untouched.
    if text.endswith("，"):
        core = text[:-1]
        if _VOXCPM2_TRAILING_DIGIT_RUN_RE.search(core) or re.search(r"\d{1,2}$", core):
            text = core + "。"
    elif not re.search(r"[。！？]$", text):
        text = _VOXCPM2_TRAILING_DIGIT_RUN_RE.sub(r"\1\2。", text)
        if re.search(r"\d{1,2}$", text):
            text = f"{text}。"
    # Hold a LONE edge '0' out of TN conversion. The lookbehind matters: in
    # "10。" the 0 is the second digit of the run - sentinel-splitting it
    # would emit "1<sentinel>。" i.e. "一零".
    if _VOXCPM2_LONE_TRAILING_ZERO_RE.search(text):
        text = text[:-2] + _VOXCPM2_ZERO_SENTINEL + "。"
    return text


def normalize_voxcpm2_text(text: str, text_normalizer: Any) -> str:
    """Run TextNormalizer when the text contains TN-trigger characters, then
    restore sentinel-held digits. A lone leading "0，" is held out of TN with
    the sentinel (normalized 零 is swallowed at the leading edge; the raw
    digit token voices a weak-but-right líng)."""
    if text and _LONE_LEADING_ZERO_RE.match(text):
        text = _VOXCPM2_ZERO_SENTINEL + text[1:]
    if text and os.environ.get("VLLM_OMNI_SPEECH_TEXT_NORMALIZE", "1") != "0" and _VOXCPM2_TN_TRIGGER_RE.search(text):
        normalized = text_normalizer.normalize(text)
        if isinstance(normalized, str):
            text = normalized
    return text.replace(_VOXCPM2_ZERO_SENTINEL, "0")
