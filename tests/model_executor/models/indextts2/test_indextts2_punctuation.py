# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for IndexTTS-2 Chinese punctuation normalisation.

This test suite validates that the *production* ``normalize_text()`` in
``vllm_omni/model_executor/models/indextts2/utils/front.py`` correctly
maps Chinese punctuation to ASCII equivalents before BPE encoding,
matching the behaviour of the official index-tts-vllm inference pipeline.

The tests import the shipped implementation (``normalize_text``,
``tokenize_by_CJK_char``, ``TextTokenizer``) rather than a hand-copied
duplicate, so regressions in the production code -- such as a removed
``TextTokenizer.__init__`` -- are caught here.  The
``SentencePieceProcessor`` and vocab-file access needed to construct a
real tokenizer are mocked; no model weights or GPU are required.

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
    pytest tests/model_executor/models/indextts2/test_indextts2_punctuation.py -v
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

# Import the *shipped* production code so the regression test covers the
# implementation that is actually merged.  See review feedback on #5590:
# hand-copied duplicates let all cases pass while the real TextTokenizer
# cannot be constructed.
from vllm_omni.model_executor.models.indextts2.utils.common import (
    tokenize_by_CJK_char,
)
from vllm_omni.model_executor.models.indextts2.utils.front import (
    TextTokenizer,
    normalize_text,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---- 1. normalize_text unit tests ----


class TestNormalizeText:
    """Verify that ``normalize_text`` correctly maps Chinese punctuation."""

    def test_all_chinese_punctuation_mapped(self):
        """Every entry in _CHAR_REP_MAP produces the expected output."""
        cases: dict[str, str] = {
            "\u3002": ".",  # 。
            "\uff01": "!",  # ！
            "\uff1f": "?",  # ？
            "\uff0c": ",",  # ，
            "\uff1a": ",",  # ：
            "\uff1b": ",",  # ；
            "\u3001": ",",  # 、
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
            "\u2014": "-",  # —  (hyphen, not space, so split_segments_by_token
            "\uff5e": "-",  # ～  can use its hyphen fallback on long inputs)
            "\u00b7": "-",  # ·
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
        assert normalize_text("你好，今天天气真好！你吃饭了吗？") == (
            "你好,今天天气真好!你吃饭了吗?"
        )

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_no_cjk_punctuation_remains(self):
        """After normalisation, no original Chinese punctuation chars remain."""
        text = "，。！？：；""''（）【】《》——～·"
        result = normalize_text(text)
        all_cjk_punc = (
            "\u3002\uff01\uff1f\uff0c\u201c\u201d\u2018\u2019"
            "\u300a\u300b\u3010\u3011\u2014\uff5e\u00b7"
        )
        for ch in all_cjk_punc:
            assert ch not in result, (
                f"{ch!r} (U+{ord(ch):04X}) should have been replaced"
            )

    def test_book_title_marks(self):
        assert normalize_text("《红楼梦》") == "'红楼梦'"

    def test_newlines_and_periods(self):
        assert normalize_text("第一行。\n第二行。") == "第一行. 第二行."

    def test_em_dash_maps_to_hyphen_not_space(self):
        """U+2014 (—) and U+FF5E (～) must map to '-' so that
        ``split_segments_by_token`` can use its hyphen fallback on long
        inputs.  Mapping them to a space removes the boundary (see review
        feedback on #5590)."""
        assert normalize_text("\u2014") == "-"
        assert normalize_text("\uff5e") == "-"


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
        assert "." in TextTokenizer.punctuation_marks_tokens

    def test_chinese_period_not_split_token(self):
        assert "。" not in TextTokenizer.punctuation_marks_tokens

    def test_chinese_exclamation_not_split_token(self):
        assert "！" not in TextTokenizer.punctuation_marks_tokens

    def test_chinese_question_not_split_token(self):
        assert "？" not in TextTokenizer.punctuation_marks_tokens

    def test_normalize_maps_period_to_split_token(self):
        """After ``normalize_text``, the period ``。`` becomes ``.``
        which IS a recognised split token."""
        assert normalize_text("。") in TextTokenizer.punctuation_marks_tokens


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
            TextTokenizer.split_segments_by_token(
                long_tokens, TextTokenizer.punctuation_marks_tokens, 120
            )

    def test_ascii_period_no_warning(self):
        """The same sequence with ASCII periods (.) splits properly
        and does NOT warn."""
        long_tokens: list[str] = list("hello" + ".world" * 121)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TextTokenizer.split_segments_by_token(
                long_tokens, TextTokenizer.punctuation_marks_tokens, 120
            )
            assert not any(
                "exceeds limit" in str(msg.message).lower() for msg in w
            ), "ASCII periods should split fine without warning"

    def test_normalized_text_no_warning(self):
        """After ``normalize_text``, ``。`` becomes ``.`` so the same
        long Chinese text splits correctly without warnings."""
        long_tokens: list[str] = list("hello" + ".world" * 121)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TextTokenizer.split_segments_by_token(
                long_tokens, TextTokenizer.punctuation_marks_tokens, 120
            )
            assert not any(
                "exceeds limit" in str(msg.message).lower() for msg in w
            ), "Normalised (。→.) text should split fine"

    def test_em_dash_long_text_uses_hyphen_fallback(self):
        """A long token sequence containing '-' (the normalised form of
        — and ～) splits on the hyphen fallback in
        ``split_segments_by_token`` and does NOT warn, because '-' is
        not in ``punctuation_marks_tokens`` and triggers the hyphen
        sub-split branch."""
        long_tokens: list[str] = list("hello" + "-world" * 121)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TextTokenizer.split_segments_by_token(
                long_tokens, TextTokenizer.punctuation_marks_tokens, 120
            )
            assert not any(
                "exceeds limit" in str(msg.message).lower() for msg in w
            ), "Hyphen (from —/～) should split via the hyphen fallback"


# ---- 5. End-to-end pipeline simulation ----


class TestEndToEndPipeline:
    """Simulate the full text-processing pipeline
    (normalize → CJK tokenize → split) to verify the fix."""

    def _simulate(self, text: str, use_normalize: bool = True) -> list[list[str]]:
        if use_normalize:
            text = normalize_text(text)
        cjk_tok = tokenize_by_CJK_char(text)
        tokens = cjk_tok.split()
        return TextTokenizer.split_segments_by_token(
            tokens, TextTokenizer.punctuation_marks_tokens, 120
        )

    def test_normalized_long_text_within_limit(self):
        """With normalisation enabled, a long Chinese text with ``。``
        stays within the segment limit."""
        text = "今天天气真好。" + "好" * 119 + "再见。"
        segments = self._simulate(text, use_normalize=True)
        for i, seg in enumerate(segments):
            assert len(seg) <= 120, f"Segment {i} exceeds limit: {len(seg)} tokens"

    def test_raw_long_text_triggers_warning(self):
        """Without normalisation, the same text triggers RuntimeWarning."""
        text = "今天天气真好。" + "好" * 119 + "再见。"
        with pytest.warns(RuntimeWarning, match="exceeds limit"):
            self._simulate(text, use_normalize=False)


# ---- 6. TextTokenizer construction & encode path ----


class TestTextTokenizerConstruction:
    """Regression tests for the shipped ``TextTokenizer`` class.

    Review feedback on #5590: a previous revision deleted
    ``TextTokenizer.__init__`` while ``IndexTTS2Tokenizer.__init__`` still
    called ``TextTokenizer(vocab_file)``, so every tokenizer load raised
    ``TypeError: TextTokenizer() takes no arguments``.  These tests
    construct the real ``TextTokenizer`` (mocking only SentencePiece and
    the vocab file) and assert that ``encode`` runs ``normalize_text`` on
    its input before hitting the BPE model.
    """

    def test_init_sets_sp_model_and_pre_tokenizers(self, tmp_path):
        vocab_file = tmp_path / "bpe.model"
        vocab_file.write_text("dummy")  # exists() must pass

        with patch(
            "vllm_omni.model_executor.models.indextts2.utils.front."
            "SentencePieceProcessor"
        ) as mock_spp_cls:
            tok = TextTokenizer(str(vocab_file))

        assert tok.vocab_file == str(vocab_file)
        assert mock_spp_cls.called  # SentencePieceProcessor(model_file=...)
        assert tok.sp_model is mock_spp_cls.return_value
        assert tok.pre_tokenizers == [tokenize_by_CJK_char]

    def test_init_raises_when_vocab_file_missing(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.model")
        with pytest.raises(ValueError, match="does not exist"):
            TextTokenizer(missing)

    def test_init_raises_when_vocab_file_none(self):
        with pytest.raises(ValueError, match="vocab_file is None"):
            TextTokenizer(None)

    def test_encode_runs_normalize_text_before_bpe(self, tmp_path):
        """``encode`` must normalise Chinese punctuation to ASCII before
        handing the text to the SentencePiece BPE model, otherwise the
        model mispronounces it (e.g. ``。`` read as ``哦``)."""
        vocab_file = tmp_path / "bpe.model"
        vocab_file.write_text("dummy")

        with patch(
            "vllm_omni.model_executor.models.indextts2.utils.front."
            "SentencePieceProcessor"
        ) as mock_spp_cls:
            mock_sp = mock_spp_cls.return_value
            mock_sp.Encode.return_value = [1, 2, 3]
            tok = TextTokenizer(str(vocab_file))

            tok.encode("你好。今天好吗？")

        # The BPE model should have received the normalised (ASCII) text,
        # not the raw Chinese punctuation.
        assert mock_sp.Encode.called
        sent_text = mock_sp.Encode.call_args.args[0]
        assert "。" not in sent_text
        assert "？" not in sent_text
        assert "." in sent_text
        assert "?" in sent_text

    def test_encode_skips_normalize_for_single_char(self, tmp_path):
        """The single-character fast path must not break; it delegates
        directly to ``sp_model.Encode``."""
        vocab_file = tmp_path / "bpe.model"
        vocab_file.write_text("dummy")

        with patch(
            "vllm_omni.model_executor.models.indextts2.utils.front."
            "SentencePieceProcessor"
        ) as mock_spp_cls:
            mock_sp = mock_spp_cls.return_value
            mock_sp.Encode.return_value = [42]
            tok = TextTokenizer(str(vocab_file))

            result = tok.encode("A")

        assert result == [42]
        assert mock_sp.Encode.called
