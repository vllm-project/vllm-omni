# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.entrypoints.openai.speech_text_splitter import SpeechTextSplitter, extract_complete_units

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestExtractCompleteUnits:
    def test_latin_sentence_needs_whitespace_or_flush(self):
        units, remainder = extract_complete_units("Hello world.", frozenset(".!?。"), flush=False)
        assert units == []
        assert remainder == "Hello world."

        units, remainder = extract_complete_units("Hello world.", frozenset(".!?。"), flush=True)
        assert units == ["Hello world."]
        assert remainder == ""

    def test_latin_sentence_with_trailing_space(self):
        units, remainder = extract_complete_units("Hello world. More", frozenset(".!?"), flush=False)
        assert units == ["Hello world."]
        assert remainder == "More"

    def test_decimal_is_not_a_sentence_boundary(self):
        units, remainder = extract_complete_units("It costs 3.14 dollars. ", frozenset(".!?"), flush=False)
        assert units == ["It costs 3.14 dollars."]
        assert remainder == ""

    def test_cjk_period_completes_without_space(self):
        units, remainder = extract_complete_units("你好。世界。", frozenset(".!?。！？"), flush=False)
        assert units == ["你好。", "世界。"]
        assert remainder == ""

    def test_indic_danda_and_question_mark(self):
        units, remainder = extract_complete_units("नमस्ते। कैसे हो?", frozenset(".!?।॥؟"), flush=True)
        assert units == ["नमस्ते।", "कैसे हो?"]
        assert remainder == ""


class TestSpeechTextSplitter:
    def test_none_buffers_until_flush(self):
        splitter = SpeechTextSplitter("none")
        assert splitter.feed("Hello world. ") == []
        assert splitter.has_buffered_text()
        assert splitter.flush() == ["Hello world."]
        assert not splitter.has_buffered_text()

    def test_sentence_mode_incremental_words(self):
        splitter = SpeechTextSplitter("sentence")
        assert splitter.feed("Hello ") == []
        assert splitter.feed("world. ") == ["Hello world."]
        assert splitter.feed("How are you?") == []
        assert splitter.flush() == ["How are you?"]

    def test_clause_mode_splits_on_comma(self):
        splitter = SpeechTextSplitter("clause")
        assert splitter.feed("Hello, world. ") == ["Hello,", "world."]

    def test_empty_flush(self):
        splitter = SpeechTextSplitter("sentence")
        assert splitter.flush() == []
