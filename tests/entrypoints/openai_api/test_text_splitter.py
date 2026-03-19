"""Tests for SentenceSplitter used in streaming TTS input."""

import pytest

from vllm_omni.entrypoints.openai.text_splitter import SentenceSplitter

pytestmark = [pytest.mark.openai, pytest.mark.speech]


class TestSentenceSplitterEnglish:
    """Tests for English sentence splitting."""

    def test_single_sentence_no_boundary(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello world")
        assert result == []
        assert splitter.buffer == "Hello world"

    def test_single_sentence_with_boundary(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello world. How are you?")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_multiple_sentences(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello. How are you? I am fine! ")
        assert len(result) == 3
        assert result[0] == "Hello."
        assert result[1] == "How are you?"
        assert result[2] == "I am fine!"

    def test_exclamation_mark(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Wow, that is great! Tell me more.")
        assert len(result) == 1
        assert result[0] == "Wow, that is great!"

    def test_question_mark(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Can you hear me? I hope so.")
        assert len(result) == 1
        assert result[0] == "Can you hear me?"


class TestSentenceSplitterChinese:
    """Tests for CJK sentence splitting."""

    def test_chinese_period(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("你好世界。你好吗")
        assert len(result) == 1
        assert result[0] == "你好世界。"

    def test_chinese_exclamation(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("太好了！谢谢你")
        assert len(result) == 1
        assert result[0] == "太好了！"

    def test_chinese_question(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("你是谁？我是小明")
        assert len(result) == 1
        assert result[0] == "你是谁？"

    def test_chinese_comma_no_split(self):
        """Chinese commas are clause-level and should not trigger a split."""
        splitter = SentenceSplitter()
        result = splitter.add_text("你好，世界")
        assert result == []
        assert splitter.buffer == "你好，世界"

    def test_chinese_semicolon_no_split(self):
        """Chinese semicolons are clause-level and should not trigger a split."""
        splitter = SentenceSplitter()
        result = splitter.add_text("第一点；第二点")
        assert result == []
        assert splitter.buffer == "第一点；第二点"

    def test_chinese_multiple(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("你好！你好吗？我很好。")
        assert len(result) == 3
        assert result[0] == "你好！"
        assert result[1] == "你好吗？"
        assert result[2] == "我很好。"


class TestSentenceSplitterMixed:
    """Tests for mixed-language sentence splitting."""

    def test_mixed_english_chinese(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello世界。How are you? ")
        assert len(result) == 2
        assert result[0] == "Hello世界。"
        assert result[1] == "How are you?"


class TestSentenceSplitterIncremental:
    """Tests for incremental (multi-chunk) text input."""

    def test_accumulation_across_chunks(self):
        splitter = SentenceSplitter()
        # First chunk: no boundary
        result1 = splitter.add_text("Hello ")
        assert result1 == []

        # Second chunk: completes a sentence
        result2 = splitter.add_text("world. How")
        assert len(result2) == 1
        assert result2[0] == "Hello world."
        assert splitter.buffer == "How"

    def test_word_by_word(self):
        splitter = SentenceSplitter()
        words = ["Hello, ", "how ", "are ", "you? ", "I ", "am ", "fine."]
        all_sentences = []
        for word in words:
            all_sentences.extend(splitter.add_text(word))

        assert len(all_sentences) == 1
        assert all_sentences[0] == "Hello, how are you?"
        # "I am fine." stays in buffer (no trailing whitespace after period)

    def test_three_chunks(self):
        splitter = SentenceSplitter()
        splitter.add_text("The quick brown ")
        splitter.add_text("fox jumps. ")
        result = splitter.add_text("Over the lazy dog. ")
        # "The quick brown fox jumps." should have been returned on second chunk
        # "Over the lazy dog." on third chunk
        assert len(result) == 1
        assert result[0] == "Over the lazy dog."


class TestSentenceSplitterFlush:
    """Tests for flush behavior."""

    def test_flush_returns_remaining(self):
        splitter = SentenceSplitter()
        splitter.add_text("Hello world")
        result = splitter.flush()
        assert result == "Hello world"
        assert splitter.buffer == ""

    def test_flush_empty_buffer(self):
        splitter = SentenceSplitter()
        result = splitter.flush()
        assert result is None

    def test_flush_after_sentence(self):
        splitter = SentenceSplitter()
        splitter.add_text("Hello world. Remaining text")
        result = splitter.flush()
        assert result == "Remaining text"

    def test_flush_whitespace_only(self):
        splitter = SentenceSplitter()
        splitter.add_text("Hello. ")
        # "Hello." extracted, buffer is " "
        result = splitter.flush()
        # Whitespace-only should return None
        assert result is None

    def test_flush_clears_buffer(self):
        splitter = SentenceSplitter()
        splitter.add_text("some text")
        splitter.flush()
        assert splitter.buffer == ""
        # Second flush should return None
        assert splitter.flush() is None


class TestSentenceSplitterEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("")
        assert result == []
        assert splitter.buffer == ""

    def test_none_like_empty(self):
        """Empty string should not affect buffer."""
        splitter = SentenceSplitter()
        splitter.add_text("Hello")
        splitter.add_text("")
        assert splitter.buffer == "Hello"

    def test_only_punctuation(self):
        splitter = SentenceSplitter()
        result = splitter.add_text(". ")
        # "." is 1 char, below default min_sentence_length of 2
        # It will be carried forward
        assert result == []

    def test_min_sentence_length(self):
        splitter = SentenceSplitter(min_sentence_length=10)
        result = splitter.add_text("Hi. Hello world. ")
        # "Hi." is 3 chars (< 10), so it gets carried to "Hello world."
        assert len(result) == 1
        assert "Hi." in result[0]
        assert "Hello world." in result[0]

    def test_short_segments_are_carried_until_long_enough(self):
        splitter = SentenceSplitter(min_sentence_length=10)
        result = splitter.add_text("Hi. Ok. Hello there. ")
        assert result == ["Hi.Ok.Hello there."]
        assert splitter.buffer == ""

    def test_min_sentence_length_zero(self):
        splitter = SentenceSplitter(min_sentence_length=0)
        result = splitter.add_text("A. B. ")
        assert len(result) == 2

    def test_no_boundary_then_flush(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello world how are you")
        assert result == []
        flushed = splitter.flush()
        assert flushed == "Hello world how are you"

    def test_consecutive_punctuation(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Really?! Yes, really. ")
        assert len(result) >= 1

    def test_reuse_after_flush(self):
        """Splitter can be reused after flush."""
        splitter = SentenceSplitter()
        splitter.add_text("First session.")
        splitter.flush()

        result = splitter.add_text("Second session. More text")
        assert len(result) == 1
        assert result[0] == "Second session."
        assert splitter.buffer == "More text"


class TestSentenceSplitterBufferLimit:
    """Tests for buffer overflow protection."""

    def test_buffer_overflow_raises(self):
        from vllm_omni.entrypoints.openai.text_splitter import _MAX_BUFFER_SIZE

        splitter = SentenceSplitter()
        # Fill buffer just under the limit
        splitter.add_text("x" * (_MAX_BUFFER_SIZE - 1))
        # One more char should exceed the limit
        with pytest.raises(ValueError, match="exceeded maximum size"):
            splitter.add_text("xx")
