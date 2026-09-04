# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.entrypoints.openai.speech_text_splitter import SpeechTextSplitter, extract_complete_units

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestExtractCompleteUnits:
    def test_ascii_sentence_needs_whitespace_or_flush(self):
        units, remainder, _ = extract_complete_units("Hello world.", frozenset(".!?。"), flush=False)
        assert units == []
        assert remainder == "Hello world."

        units, remainder, _ = extract_complete_units("Hello world.", frozenset(".!?。"), flush=True)
        assert units == ["Hello world."]
        assert remainder == ""

    def test_ascii_sentence_with_trailing_space(self):
        units, remainder, _ = extract_complete_units("Hello world. More", frozenset(".!?"), flush=False)
        assert units == ["Hello world."]
        assert remainder == "More"

    def test_decimal_is_not_a_sentence_boundary(self):
        units, remainder, _ = extract_complete_units("It costs 3.14 dollars. ", frozenset(".!?"), flush=False)
        assert units == ["It costs 3.14 dollars."]
        assert remainder == ""

    def test_thousands_separator_is_not_a_clause_boundary(self):
        units, _, _ = extract_complete_units("It is 1,000 rupees, sir. ", frozenset(".!?,;"), flush=False)
        assert units == ["It is 1,000 rupees,", "sir."]

    def test_punctuation_run_stays_one_unit(self):
        units, _, _ = extract_complete_units("Wait... Really?! Yes.", frozenset(".!?"), flush=True)
        assert units == ["Wait...", "Really?!", "Yes."]

    def test_closing_quote_stays_with_its_sentence(self):
        units, _, _ = extract_complete_units('He said "Hello." Bye.', frozenset(".!?"), flush=True)
        assert units == ['He said "Hello."', "Bye."]

    def test_abbreviations_and_initials_do_not_split(self):
        units, _, _ = extract_complete_units("Dr. Smith, e.g. J. R. Tolkien. Next.", frozenset(".!?"), flush=True)
        assert units == ["Dr. Smith, e.g. J. R. Tolkien.", "Next."]

    def test_cjk_period_completes_once_the_next_char_is_known(self):
        # The trailing 。 waits: a closing delimiter could still follow it.
        units, remainder, _ = extract_complete_units("你好。世界。", frozenset(".!?。！？"), flush=False)
        assert units == ["你好。"]
        assert remainder == "世界。"

        units, remainder, _ = extract_complete_units("你好。世界。", frozenset(".!?。！？"), flush=True)
        assert units == ["你好。", "世界。"]
        assert remainder == ""

    def test_indic_danda_and_question_mark(self):
        units, remainder, _ = extract_complete_units("नमस्ते। कैसे हो?", frozenset(".!?।॥؟"), flush=True)
        assert units == ["नमस्ते।", "कैसे हो?"]
        assert remainder == ""

    def test_scan_cursor_skips_already_examined_text(self):
        units, remainder, scan = extract_complete_units("no terminator yet", frozenset(".!?"), flush=False)
        assert units == []
        assert remainder == "no terminator yet"
        assert scan == len(remainder)

        units, remainder, scan = extract_complete_units(
            "no terminator yet. Next", frozenset(".!?"), flush=False, scan_from=scan
        )
        assert units == ["no terminator yet."]
        assert remainder == "Next"


class TestSpeechTextSplitter:
    def test_none_buffers_until_flush(self):
        splitter = SpeechTextSplitter("none")
        assert splitter.feed("Hello world. ") == []
        assert splitter.flush() == ["Hello world."]
        assert splitter.flush() == []

    def test_sentence_mode_incremental_words(self):
        splitter = SpeechTextSplitter("sentence")
        assert splitter.feed("Hello ") == []
        assert splitter.feed("world. ") == ["Hello world."]
        assert splitter.feed("How are you?") == []
        assert splitter.flush() == ["How are you?"]

    def test_sentence_mode_resumes_scan_across_feeds(self):
        splitter = SpeechTextSplitter("sentence")
        assert splitter.feed("It costs 3") == []
        assert splitter.feed(".") == []
        assert splitter.feed("14 dollars. ") == ["It costs 3.14 dollars."]

    def test_clause_mode_splits_on_comma(self):
        splitter = SpeechTextSplitter("clause")
        assert splitter.feed("Hello, world. ") == ["Hello,", "world."]

    def test_empty_flush(self):
        splitter = SpeechTextSplitter("sentence")
        assert splitter.flush() == []
