# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py
#
# Copyright 2026 The Alibaba Qwen team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Qwen3 forced-aligner text/timestamp processor.

This is the model-specific half of upstream's ``Qwen3ForceAlignProcessor``:
it turns text into the aligner's word units and prompt, repairs the predicted
timestamp bins, and resolves the marker token id. The generic "run a vLLM
pooling model" half lives in :mod:`vllm_omni.utils.forced_aligner`, which
calls into this module.

Keeping the Qwen-specific pieces here marks the seam for the model-agnostic
aligner the issue asks for: a different aligner family would supply its own
processor exposing the same small surface — :func:`segment_words`,
:func:`build_prompt`, :func:`fix_timestamp`, :func:`resolve_timestamp_token_id`.

Word segmentation prefers qwen_asr's official ``Qwen3ForceAlignProcessor``
when installed (full multilingual fidelity, incl. Japanese/Korean) and
otherwise uses the faithful port below: exact for whitespace-delimited and
Chinese-mixed text; Japanese/Korean degrade to whitespace splitting.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)


# Prompt tokens for the Qwen3 forced aligner. Baked in (not config) because the
# prompt template and the word segmentation are inherently Qwen-specific: a
# different aligner family needs a different processor module regardless.
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
TIMESTAMP_TOKEN = "<timestamp>"

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
)

# Map common language codes to the names qwen_asr's processor expects.
_LANG_ALIASES = {
    "ja": "japanese",
    "jp": "japanese",
    "jpn": "japanese",
    "ko": "korean",
    "kr": "korean",
    "kor": "korean",
}


def build_prompt(words: list[str]) -> str:
    """Wrap segmented words in the Qwen3 aligner prompt template.

    Each word gets two trailing ``<timestamp>`` markers (start + end); the
    model classifies each marker into an audio time bin.
    """
    if not words:
        # Pad with one timestamp so the decoder always has something to
        # read; an empty result still surfaces as "[]" upstream.
        body = TIMESTAMP_TOKEN
    else:
        body = f"{TIMESTAMP_TOKEN}{TIMESTAMP_TOKEN}".join(words) + f"{TIMESTAMP_TOKEN}{TIMESTAMP_TOKEN}"
    return f"<|im_start|>user\n{AUDIO_PLACEHOLDER}{body}<|im_end|>\n<|im_start|>assistant\n"


# --- word segmentation (port of Qwen3ForceAlignProcessor) ----------------
#
# The marker count fed to the model must match the segmentation the aligner
# was trained with, or the decode drifts and returns empty.


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _is_kept_char(ch: str) -> bool:
    """Keep letters/numbers (any script) and the apostrophe; drop punctuation."""
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _clean_token(token: str) -> str:
    return "".join(ch for ch in token if _is_kept_char(ch))


def _split_segment_with_chinese(seg: str) -> list[str]:
    """Split a cleaned segment so each CJK char is its own token."""
    tokens: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in seg:
        if _is_cjk_char(ch):
            flush()
            tokens.append(ch)
        else:
            buf.append(ch)
    flush()
    return tokens


def _tokenize_space_lang(text: str) -> list[str]:
    """Whitespace + Chinese-mixed segmentation (Qwen3ForceAlignProcessor port).

    Splits on whitespace, strips punctuation from each segment (keeping
    cross-script letters/digits and the apostrophe), then peels off CJK
    characters as individual tokens.
    """
    tokens: list[str] = []
    for seg in text.split():
        cleaned = _clean_token(seg)
        if cleaned:
            tokens.extend(_split_segment_with_chinese(cleaned))
    return tokens


# Cached qwen_asr processor (or a sentinel that it is unavailable). Importing
# it eagerly pulls nagisa + the Qwen3ASR transformers backend, so we only try
# once, lazily, and remember the outcome.
_official_processor: Any = None
_official_processor_unavailable = False


def _get_official_processor() -> Any | None:
    global _official_processor, _official_processor_unavailable
    if _official_processor is not None:
        return _official_processor
    if _official_processor_unavailable:
        return None
    try:
        from qwen_asr.inference.qwen3_forced_aligner import (  # type: ignore[import-not-found]
            Qwen3ForceAlignProcessor,
        )

        _official_processor = Qwen3ForceAlignProcessor()
    except Exception:  # noqa: BLE001 - any import/init error means "use the port"
        _official_processor_unavailable = True
        logger.info(
            "qwen_asr not available; using the built-in word segmentation "
            "(exact for whitespace/Chinese-mixed text; Japanese/Korean fall "
            "back to whitespace splitting). Install qwen-asr for full fidelity."
        )
        return None
    return _official_processor


def segment_words(text: str, language: str | None = None) -> list[str]:
    """Split ``text`` into the aligner's word units.

    Prefers qwen_asr's official processor (full multilingual fidelity); falls
    back to the built-in port otherwise. ``language`` follows the official
    naming — ``"japanese"`` / ``"korean"`` (or codes like ``ja`` / ``ko``)
    trigger language-specific tokenisers; anything else (incl. ``None`` /
    ``"auto"``) uses the whitespace + Chinese-mixed path.
    """
    lang = (language or "auto").strip().lower()
    lang = _LANG_ALIASES.get(lang, lang)
    processor = _get_official_processor()
    if processor is not None:
        try:
            word_list, _ = processor.encode_timestamp(text, lang)
            return list(word_list)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Official Qwen3ForceAlignProcessor failed for language=%r; falling back to built-in segmentation.",
                lang,
                exc_info=True,
            )
    if lang in ("japanese", "korean"):
        logger.warning(
            "language=%r needs qwen_asr (nagisa/soynlp) for faithful word "
            "segmentation; falling back to whitespace splitting.",
            lang,
        )
    return _tokenize_space_lang(text)


def fix_timestamp(values: list[int]) -> list[int]:
    """Repair non-monotonic timestamp bins via LIS + interpolation.

    The model occasionally emits out-of-order bins; this snaps anomalies back
    onto the longest non-decreasing subsequence (interpolating longer runs)
    so paired start/end markers stay ordered. Port of
    ``Qwen3ForceAlignProcessor.fix_timestamp`` (Apache-2.0).
    """
    n = len(values)
    if n == 0:
        return []

    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if values[j] <= values[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    idx = dp.index(max(dp))
    lis: list[int] = []
    while idx != -1:
        lis.append(idx)
        idx = parent[idx]
    is_normal = [False] * n
    for k in lis:
        is_normal[k] = True

    result = [float(v) for v in values]
    i = 0
    while i < n:
        if is_normal[i]:
            i += 1
            continue
        j = i
        while j < n and not is_normal[j]:
            j += 1
        count = j - i
        left = next((result[k] for k in range(i - 1, -1, -1) if is_normal[k]), None)
        right = next((result[k] for k in range(j, n) if is_normal[k]), None)
        if count <= 2:
            for k in range(i, j):
                if left is None:
                    result[k] = right
                elif right is None:
                    result[k] = left
                else:
                    result[k] = left if (k - (i - 1)) <= (j - k) else right
        elif left is not None and right is not None:
            step = (right - left) / (count + 1)
            for k in range(i, j):
                result[k] = left + step * (k - i + 1)
        elif left is not None:
            for k in range(i, j):
                result[k] = left
        elif right is not None:
            for k in range(i, j):
                result[k] = right
        i = j

    return [int(r) for r in result]


def resolve_timestamp_token_id(tokenizer: Any, timestamp_token: str = TIMESTAMP_TOKEN) -> int:
    """Look up the integer id of the timestamp special token."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise RuntimeError("Aligner tokenizer has no convert_tokens_to_ids method.")
    tid = convert(timestamp_token)
    if isinstance(tid, list):
        tid = tid[0] if tid else None
    if tid is None or (isinstance(tid, int) and tid < 0):
        raise RuntimeError(
            f"Aligner tokenizer does not recognise {timestamp_token!r} (got id={tid}). "
            "Check the model card; the marker token may use a different name."
        )
    return int(tid)


def _reset_for_tests() -> None:
    """Drop the cached official-processor probe so the next call re-checks."""
    global _official_processor, _official_processor_unavailable
    _official_processor = None
    _official_processor_unavailable = False
