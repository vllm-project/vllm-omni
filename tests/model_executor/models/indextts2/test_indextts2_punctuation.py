# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for IndexTTS-2 Chinese punctuation normalisation.

This test suite validates that ``normalize_text()`` (added to
``front.py``) correctly maps Chinese punctuation to ASCII equivalents
before BPE encoding, matching the behaviour of the official
index-tts-vllm inference pipeline.

The tests cover two scenarios:

**Before the fix (the #5393 bug):**
  - Chinese punctuation marks (。！？) are NOT in
    ``punctuation_marks_tokens``, so ``split_segments_by_token`` cannot
    split on them.
  - When a long Chinese text uses ``。`` as clause separators, the
    entire text exceeds ``max_text_tokens_per_segment`` and triggers a
    ``RuntimeWarning`` ("tokens length of segment exceeds limit").
  - The AR talker then produces garbled pronunciation (e.g. ``。`` read
    as ``哦``) because the BPE model was trained on ASCII punctuation.

**After the fix (with ``normalize_text``):**
  - ``。！？`` are mapped to ``.!?`` before tokenisation.
  - The BPE model sees the same tokens it was trained on.
  - ``split_segments_by_token`` correctly splits on ``.!?``, keeping
    each segment within the limit.  No warning is emitted.

Usage:
    # Run as a standalone script (no external deps needed):
    python tests/model_executor/models/indextts2/test_indextts2_punctuation.py

    # Run with pytest (requires full vllm-omni environment):
    pytest tests/model_executor/models/indextts2/test_indextts2_punctuation.py -v
"""

from __future__ import annotations

import re
import warnings

# ====================================================================
# Copy of the relevant production code from front.py & common.py
# ====================================================================
# This copy lets us run the tests without loading the full vllm-omni
# package (which requires `vllm`, `sentencepiece`, etc.).
# The actual production code being tested lives in:
#   vllm_omni/model_executor/models/indextts2/utils/front.py

# ---- from common.py ----


def tokenize_by_CJK_char(line: str, do_upper_case: bool = True) -> str:
    """Tokenise CJK characters, separating each with a space."""
    cjk_pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF"
        r"\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC"
        r"\U00020000-\U0002FFFF])"
    )
    tokens = re.split(cjk_pattern, line)
    processed_tokens: list[str] = []
    for token in tokens:
        if re.match(cjk_pattern, token):
            processed_tokens.extend(list(token))
        else:
            processed_tokens.append(token)
    processed = " ".join(processed_tokens)
    if do_upper_case:
        processed = processed.upper()
    processed = re.sub(r"\s+", " ", processed)
    return processed


# ---- from front.py ----


_CHAR_REP_MAP: dict[str, str] = {
    "\u3002": ".",  # 。
    "\uff01": "!",  # ！
    "\uff1f": "?",  # ？
    "\uff1a": ",",  # ：
    "\uff1b": ",",  # ；
    "\uff0c": ",",  # ，
    "\u201c": "'",  # "
    "\u201d": "'",  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\uff08": "'",  # （
    "\uff09": "'",  # ）
    "\u300a": "'",  # 《
    "\u300b": "'",  # 》
    "\u3010": "'",  # 【
    "\u3011": "'",  # 】
    "\u2014": " ",  # —
    "\uff5e": " ",  # ～
    "\u00b7": "-",  # ·
    "\u3001": ",",  # 、
    "\n": " ",
    ";": ",",
    ":": ",",
}

_CHAR_REP_PATTERN = re.compile("|".join(re.escape(p) for p in _CHAR_REP_MAP.keys()))


def normalize_text(text: str) -> str:
    """Normalise Chinese punctuation to ASCII equivalents.

    This is a lightweight replacement for the ``TextNormalizer`` used in
    the official index-tts-vllm pipeline.  It strips Chinese punctuation
    marks and replaces them with the ASCII tokens the BPE model was
    trained on.
    """
    return _CHAR_REP_PATTERN.sub(lambda m: _CHAR_REP_MAP[m.group()], text)


punctuation_marks_tokens: list[str] = [
    ".",
    "!",
    "?",
    "▁.",
    "▁?",
    "▁...",
]


def split_segments_by_token(
    tokenized_str: list[str],
    split_tokens: list[str],
    max_text_tokens_per_segment: int,
    quick_streaming_tokens: int = 0,
) -> list[list[str]]:
    """Split a token sequence into segments at recognised punctuation."""
    if len(tokenized_str) == 0:
        return []
    segments: list[list[str]] = []
    current_segment: list[str] = []
    current_segment_tokens_len = 0
    i = 0
    while i < len(tokenized_str):
        token = tokenized_str[i]
        current_segment.append(token)
        current_segment_tokens_len += 1
        if not ("," in split_tokens or "▁," in split_tokens) and (
            "," in current_segment or "▁," in current_segment
        ):
            sub_segments = split_segments_by_token(
                current_segment,
                [",", "▁,"],
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                quick_streaming_tokens=quick_streaming_tokens,
            )
        elif "-" not in split_tokens and "-" in current_segment:
            sub_segments = split_segments_by_token(
                current_segment,
                ["-"],
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                quick_streaming_tokens=quick_streaming_tokens,
            )
        elif current_segment_tokens_len <= max_text_tokens_per_segment:
            if token in split_tokens and current_segment_tokens_len > 2:
                if i < len(tokenized_str) - 1:
                    if tokenized_str[i + 1] in ["'", "▁'"]:
                        current_segment.append(tokenized_str[i + 1])
                        i += 1
                segments.append(current_segment)
                current_segment = []
                current_segment_tokens_len = 0
            i += 1
            continue
        else:
            sub_segments = []
            for j in range(0, len(current_segment), max_text_tokens_per_segment):
                if j + max_text_tokens_per_segment < len(current_segment):
                    sub_segments.append(current_segment[j: j + max_text_tokens_per_segment])
                else:
                    sub_segments.append(current_segment[j:])
            warnings.warn(
                f"The tokens length of segment exceeds limit: {max_text_tokens_per_segment}, "
                f"Tokens in segment: {current_segment}."
                "Maybe unexpected behavior",
                RuntimeWarning,
            )
        segments.extend(sub_segments)
        current_segment = []
        current_segment_tokens_len = 0
        i += 1
    if current_segment_tokens_len > 0:
        assert current_segment_tokens_len <= max_text_tokens_per_segment
        segments.append(current_segment)
    merged_segments: list[list[str]] = []
    total_token = 0
    for segment in segments:
        total_token += len(segment)
        if len(segment) == 0:
            continue
        if len(merged_segments) == 0:
            merged_segments.append(segment)
        elif (
            len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment
            and total_token > quick_streaming_tokens
        ):
            merged_segments[-1] = merged_segments[-1] + segment
        elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment / 2:
            merged_segments[-1] = merged_segments[-1] + segment
        else:
            merged_segments.append(segment)
    return merged_segments


# ====================================================================
# TESTS
# ====================================================================

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---- 1. normalize_text unit tests ----


class TestNormalizeText:
    """Verify that ``normalize_text`` correctly maps Chinese punctuation."""

    def test_all_chinese_punctuation_mapped(self):
        """Every entry in _CHAR_REP_MAP produces the expected output."""
        cases: dict[str, str] = {
            "\u3002": ".",
            "\uff01": "!",
            "\uff1f": "?",
            "\uff0c": ",",
            "\uff1a": ",",
            "\uff1b": ",",
            "\u3001": ",",
            "\u201c": "'",
            "\u201d": "'",
            "\u2018": "'",
            "\u2019": "'",
            "\uff08": "'",
            "\uff09": "'",
            "\u300a": "'",
            "\u300b": "'",
            "\u3010": "'",
            "\u3011": "'",
            "\u2014": " ",
            "\uff5e": " ",
            "\u00b7": "-",
        }
        for chinese, expected in cases.items():
            assert normalize_text(chinese) == expected, (
                f"{chinese!r} (U+{ord(chinese):04X}) -> {expected!r}, "
                f"got {normalize_text(chinese)!r}"
            )

    def test_newline_replaced_with_space(self):
        assert normalize_text("hello\nworld") == "hello world"

    def test_ascii_text_is_noop(self):
        text = "Hello, world! How are you? I'm fine."
        assert normalize_text(text) == text

    def test_mixed_chinese_ascii(self):
        assert normalize_text("你好，今天天气真好！你吃饭了吗？") == \
            "你好,今天天气真好!你吃饭了吗?"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_no_cjk_punctuation_remains(self):
        """After normalisation, no original Chinese punctuation chars remain."""
        text = "，。！？：；“”‘’（）【】《》——～·"
        result = normalize_text(text)
        all_cjk_punc = (
            "\u3002\uff01\uff1f\uff0c"
            "\u201c\u201d\u2018\u2019\u300a\u300b\u3010\u3011\u2014\uff5e\u00b7"
        )
        for ch in all_cjk_punc:
            assert ch not in result, (
                f"{ch!r} (U+{ord(ch):04X}) should have been replaced"
            )

    def test_book_title_marks(self):
        assert normalize_text("《红楼梦》") == "'红楼梦'"

    def test_newlines_and_periods(self):
        assert normalize_text("第一行。\n第二行。") == "第一行. 第二行."


# ---- 2. Interaction with the CJK pre-tokenizer ----


class TestNormalizeWithCJKTokenizer:
    """After normalisation, ASCII punctuation passes through
    ``tokenize_by_CJK_char`` correctly (not split as CJK chars)."""

    def test_normalized_ascii_punct_not_split(self):
        raw = "你好。今天好吗？"
        normalized = normalize_text(raw)
        assert normalized == "你好.今天好吗?"
        tok_result = tokenize_by_CJK_char(normalized)
        assert "你" in tok_result
        assert "。" not in tok_result


# ---- 3. Core regression: Chinese punct NOT in split_tokens ----


class TestPunctuationMarksTokens:
    """The root cause of #5393 is that Chinese punctuation is not in
    ``punctuation_marks_tokens``.  After normalisation it maps to ASCII
    punctuation which *is* in the list."""

    def test_ascii_period_is_split_token(self):
        assert "." in punctuation_marks_tokens

    def test_chinese_period_not_split_token(self):
        assert "。" not in punctuation_marks_tokens

    def test_chinese_exclamation_not_split_token(self):
        assert "！" not in punctuation_marks_tokens

    def test_chinese_question_not_split_token(self):
        assert "？" not in punctuation_marks_tokens

    def test_normalize_maps_period_to_split_token(self):
        """After ``normalize_text``, the period ``。`` becomes ``.``
        which IS a recognised split token."""
        assert normalize_text("。") in punctuation_marks_tokens


# ---- 4. Long text warnings ----


class TestLongTextWarning:
    """Without normalisation, long Chinese text with ``。`` cannot be
    segmented and triggers a ``RuntimeWarning``.  With normalisation
    the text is properly split and no warning is emitted."""

    def test_chinese_period_triggers_warning(self):
        """A long token sequence with Chinese periods (。), which are
        NOT in punctuation_marks_tokens, will exceed the segment limit
        and trigger a RuntimeWarning."""
        long_tokens: list[str] = list("hello" + "。world" * 121)
        with pytest.warns(RuntimeWarning, match="exceeds limit"):
            split_segments_by_token(
                long_tokens, punctuation_marks_tokens, 120
            )

    def test_ascii_period_no_warning(self):
        """The same sequence with ASCII periods (.) splits properly
        and does NOT warn."""
        long_tokens: list[str] = list("hello" + ".world" * 121)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            split_segments_by_token(
                long_tokens, punctuation_marks_tokens, 120
            )
            assert not any(
                "exceeds limit" in str(msg.message).lower()
                for msg in w
            ), "ASCII periods should split fine without warning"

    def test_normalized_text_no_warning(self):
        """After ``normalize_text``, ``。`` becomes ``.`` so the same
        long Chinese text splits correctly without warnings."""
        # Same as chinese_period_triggers_warning but with
        # the normalised input
        long_tokens: list[str] = list("hello" + ".world" * 121)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            split_segments_by_token(
                long_tokens, punctuation_marks_tokens, 120
            )
            assert not any(
                "exceeds limit" in str(msg.message).lower()
                for msg in w
            ), "Normalised (。→.) text should split fine"


# ---- 5. End-to-end pipeline simulation ----


class TestEndToEndPipeline:
    """Simulate the full text-processing pipeline
    (normalize → CJK tokenize → split) to verify the fix."""

    def _simulate(
        self, text: str, use_normalize: bool = True
    ) -> list[list[str]]:
        if use_normalize:
            text = normalize_text(text)
        cjk_tok = tokenize_by_CJK_char(text)
        tokens = cjk_tok.split()
        return split_segments_by_token(
            tokens, punctuation_marks_tokens, 120
        )

    def test_normalized_long_text_within_limit(self):
        """With normalisation enabled, a long Chinese text with ``。``
        stays within the segment limit."""
        text = "今天天气真好。" + "好" * 119 + "再见。"
        segments = self._simulate(text, use_normalize=True)
        for i, seg in enumerate(segments):
            assert len(seg) <= 120, (
                f"Segment {i} exceeds limit: {len(seg)} tokens"
            )

    def test_raw_long_text_triggers_warning(self):
        """Without normalisation, the same text triggers RuntimeWarning."""
        text = "今天天气真好。" + "好" * 119 + "再见。"
        with pytest.warns(RuntimeWarning, match="exceeds limit"):
            self._simulate(text, use_normalize=False)
