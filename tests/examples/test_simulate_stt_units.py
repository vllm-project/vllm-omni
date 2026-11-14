# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for simulate_stt_units() in streaming_speech_client.py.

``--simulate-stt`` only means anything if the text is actually sent piece by
piece. CJK is not written with spaces, so the space-splitting rule sent the
whole text as one ``input.text`` frame -- a silent degradation with no error.
These tests pin both splitting rules and the round-trip invariant that the
pieces reassemble into the original text.
"""

import sys
from pathlib import Path

import pytest

# Import the helper from the example script.
_examples_dir = str(
    Path(__file__).parent.parent.parent / "examples" / "online_serving" / "text_to_speech" / "qwen3_tts"
)
sys.path.insert(0, _examples_dir)
from streaming_speech_client import simulate_stt_units  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSpaceDelimited:
    """Text without CJK keeps the original space-splitting rule."""

    def test_words_split_on_space(self):
        assert simulate_stt_units("Hello world. How are you?") == [
            "Hello ",
            "world. ",
            "How ",
            "are ",
            "you?",
        ]

    def test_separator_space_kept_on_all_but_last_word(self):
        assert simulate_stt_units("a b c") == ["a ", "b ", "c"]

    def test_single_word(self):
        assert simulate_stt_units("hello") == ["hello"]


class TestCJK:
    """Text with CJK splits per CJK character; non-CJK runs stay together."""

    def test_han_single_characters(self):
        assert simulate_stt_units("今天天气真好。") == [
            "今",
            "天",
            "天",
            "气",
            "真",
            "好",
            "。",
        ]

    def test_han_is_not_sent_as_one_message(self):
        """The bug this PR fixes: pure CJK used to be a single frame."""
        text = "今天天气真好。我们出去走走吧。"
        assert len(simulate_stt_units(text)) == 15
        assert "".join(simulate_stt_units(text)) == text

    def test_latin_run_kept_together(self):
        assert simulate_stt_units("Hello 你好 world") == ["Hello ", "你", "好", " world"]

    def test_hiragana(self):
        assert simulate_stt_units("こんにちは") == ["こ", "ん", "に", "ち", "は"]

    def test_katakana(self):
        assert simulate_stt_units("テスト") == ["テ", "ス", "ト"]

    def test_hangul(self):
        assert simulate_stt_units("안녕하세요") == ["안", "녕", "하", "세", "요"]

    def test_non_bmp_han(self):
        """U+20000 is outside the BMP, so a range like 一..鿿 missed it."""
        assert simulate_stt_units("a𠀀b") == ["a", "𠀀", "b"]


class TestRoundTrip:
    """The pieces must reassemble into the original text in every case."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "hello",
            "Hello world. How are you?",
            "今天天气真好。",
            "今天天气真好。我们出去走走吧。",
            "Hello 你好 world",
            "こんにちは世界",
            "テスト",
            "안녕하세요",
            "a𠀀b",
            "版本v2.1已发布，详见readme文件。",
            "digits 123, punctuation!?",
        ],
    )
    def test_join_reproduces_input(self, text):
        assert "".join(simulate_stt_units(text)) == text
