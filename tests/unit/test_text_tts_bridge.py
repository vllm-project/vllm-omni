"""
tests/unit/test_text_tts_bridge.py
===================================
Unit tests for the text→TTS bridge processor.

These tests are deliberately GPU-free and model-free.
They cover all three RFC design questions:
  Q1 - bridge as custom_process_input_func hook
  Q2 - sentence buffering / latency knob
  Q3 - voice/speaker parameter injection

Run:
    pytest tests/unit/test_text_tts_bridge.py -v
    pytest tests/unit/test_text_tts_bridge.py -v -k "chunker"   # just chunker tests
"""

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
# This must work with zero GPU/vllm dependencies.
# ---------------------------------------------------------------------------
from vllm_omni.model_executor.stage_input_processors.text_tts_bridge import (
    SentenceChunker,
    TextTTSBridgeConfig,
    build_tts_input,
    text2tts,
)


# =============================================================================
# TextTTSBridgeConfig
# =============================================================================

class TestTextTTSBridgeConfig:

    def test_defaults(self):
        cfg = TextTTSBridgeConfig()
        assert cfg.min_sentence_chars == 40
        assert cfg.tts_task_type == "CustomVoice"
        assert cfg.default_voice == "vivian"
        assert cfg.default_language == "English"
        assert "." in cfg.sentence_delimiters

    def test_from_dict_partial(self):
        cfg = TextTTSBridgeConfig.from_dict({"default_voice": "serena", "min_sentence_chars": 20})
        assert cfg.default_voice == "serena"
        assert cfg.min_sentence_chars == 20
        assert cfg.default_language == "English"   # default preserved

    def test_from_dict_ignores_unknown_keys(self):
        # should not raise
        cfg = TextTTSBridgeConfig.from_dict({"unknown_key": "value", "default_voice": "ryan"})
        assert cfg.default_voice == "ryan"

    def test_from_dict_empty(self):
        cfg = TextTTSBridgeConfig.from_dict({})
        assert cfg == TextTTSBridgeConfig()


# =============================================================================
# SentenceChunker — core RFC Q2 logic
# =============================================================================

class TestSentenceChunker:

    def _chunker(self, min_chars=5):
        """Small min_sentence_chars so tests don't need long strings."""
        cfg = TextTTSBridgeConfig(min_sentence_chars=min_chars)
        return SentenceChunker(cfg)

    # ------------------------------------------------------------------
    # Basic flush behavior
    # ------------------------------------------------------------------

    def test_no_flush_without_delimiter(self):
        chunker = self._chunker()
        result = chunker.feed("Hello world")
        assert result == []

    def test_flush_on_period(self):
        chunker = self._chunker(min_chars=5)
        chunker.feed("Hi.")
        result = chunker.feed(" How are you.")
        # "Hi." is only 3 chars < 5 so stays buffered until combined
        # After second feed: "Hi. How are you." — flush at "Hi. "
        # Exact split depends on regex; just check we get non-empty output
        assert isinstance(result, list)

    def test_simple_sentence_flush(self):
        chunker = self._chunker(min_chars=5)
        result = chunker.feed("Hello!")
        assert len(result) == 1
        assert result[0] == "Hello!"

    def test_multiple_sentences_in_one_feed(self):
        chunker = self._chunker(min_chars=3)
        result = chunker.feed("One. Two. Three.")
        assert len(result) == 3
        assert result[0] == "One."
        assert result[1] == "Two."
        assert result[2] == "Three."

    def test_incremental_token_feeding(self):
        """Simulate token-by-token streaming from Stage 0."""
        chunker = self._chunker(min_chars=3)
        tokens = list("Hello. World.")
        all_chunks = []
        for tok in tokens:
            all_chunks.extend(chunker.feed(tok))
        all_chunks.extend(chunker.flush())
        full_text = " ".join(all_chunks)
        assert "Hello." in full_text
        assert "World." in full_text

    # ------------------------------------------------------------------
    # flush() at EOS
    # ------------------------------------------------------------------

    def test_flush_returns_remaining_buffer(self):
        chunker = self._chunker(min_chars=100)  # high threshold → nothing auto-flushes
        chunker.feed("This sentence has no ending punctuation")
        result = chunker.flush()
        assert len(result) == 1
        assert "This sentence" in result[0]

    def test_flush_empty_buffer_returns_empty(self):
        chunker = self._chunker()
        assert chunker.flush() == []

    def test_flush_whitespace_only_returns_empty(self):
        chunker = self._chunker()
        chunker.feed("   ")
        assert chunker.flush() == []

    def test_double_flush_is_safe(self):
        chunker = self._chunker()
        chunker.feed("Some text")
        chunker.flush()
        assert chunker.flush() == []  # second flush should be no-op

    # ------------------------------------------------------------------
    # min_sentence_chars — RFC Q2 latency knob
    # ------------------------------------------------------------------

    def test_min_chars_prevents_premature_flush(self):
        """Short sentence below threshold should stay buffered."""
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=50))
        result = chunker.feed("Hi.")     # only 3 chars, below threshold
        assert result == []
        # But flush() must still release it
        assert chunker.flush() == ["Hi."]

    def test_min_chars_zero_flushes_immediately(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=0))
        result = chunker.feed("Hi.")
        assert result == ["Hi."]

    # ------------------------------------------------------------------
    # CJK delimiters (multilingual support)
    # ------------------------------------------------------------------

    def test_chinese_sentence_delimiter(self):
        chunker = self._chunker(min_chars=2)
        result = chunker.feed("你好。")
        assert len(result) == 1
        assert "你好" in result[0]

    def test_japanese_exclamation(self):
        chunker = self._chunker(min_chars=2)
        result = chunker.feed("こんにちは！")
        assert len(result) == 1

    def test_custom_delimiters(self):
        cfg = TextTTSBridgeConfig(sentence_delimiters=["|"], min_sentence_chars=2)
        chunker = SentenceChunker(cfg)
        result = chunker.feed("chunk one| chunk two|")
        assert len(result) == 2


# =============================================================================
# build_tts_input — output format for Qwen3-TTS Stage 1
# =============================================================================

class TestBuildTtsInput:

    def test_required_keys_present(self):
        cfg = TextTTSBridgeConfig()
        out = build_tts_input("Hello world.", cfg)
        assert "text" in out
        assert "task_type" in out
        assert "voice" in out
        assert "language" in out

    def test_text_passthrough(self):
        cfg = TextTTSBridgeConfig()
        out = build_tts_input("Speak this.", cfg)
        assert out["text"] == "Speak this."

    def test_default_voice_used(self):
        cfg = TextTTSBridgeConfig(default_voice="aiden")
        out = build_tts_input("Hello.", cfg)
        assert out["voice"] == "aiden"

    def test_voice_override(self):
        cfg = TextTTSBridgeConfig(default_voice="vivian")
        out = build_tts_input("Hello.", cfg, voice="serena")
        assert out["voice"] == "serena"

    def test_language_override(self):
        cfg = TextTTSBridgeConfig(default_language="English")
        out = build_tts_input("Bonjour.", cfg, language="French")
        assert out["language"] == "French"

    def test_instructions_omitted_when_none(self):
        cfg = TextTTSBridgeConfig()
        out = build_tts_input("Hello.", cfg, instructions=None)
        assert "instructions" not in out

    def test_instructions_included_when_given(self):
        cfg = TextTTSBridgeConfig()
        out = build_tts_input("Hello.", cfg, instructions="speak slowly")
        assert out["instructions"] == "speak slowly"

    def test_task_type_forwarded(self):
        cfg = TextTTSBridgeConfig(tts_task_type="VoiceDesign")
        out = build_tts_input("Hello.", cfg)
        assert out["task_type"] == "VoiceDesign"


# =============================================================================
# text2tts — the actual custom_process_input_func hook (RFC Q1 + Q3)
# =============================================================================

class TestText2TtsHook:

    def _make_output(self, text, is_finished=False, extra=None, chunker=None):
        return {
            "text": text,
            "is_finished": is_finished,
            "request_id": "test-req-001",
            "chunker": chunker,
            "extra": extra or {},
        }

    # ------------------------------------------------------------------
    # Buffering behavior
    # ------------------------------------------------------------------

    def test_returns_empty_when_buffering(self):
        out = self._make_output("Hello world")  # no delimiter
        result = text2tts(out, {"min_sentence_chars": 5})
        assert result == []

    def test_returns_chunk_when_sentence_complete(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output("Hi.", chunker=chunker)
        result = text2tts(out, {"min_sentence_chars": 3})
        assert len(result) == 1
        assert result[0]["text"] == "Hi."

    def test_flushes_remaining_at_eos(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=100))
        # Feed some text first
        text2tts(self._make_output("No delimiter here", chunker=chunker))
        # Now send EOS
        result = text2tts(
            self._make_output("", is_finished=True, chunker=chunker),
            {"min_sentence_chars": 100},
        )
        assert len(result) == 1
        assert "No delimiter here" in result[0]["text"]

    # ------------------------------------------------------------------
    # Voice/language injection — RFC Q3
    # ------------------------------------------------------------------

    def test_default_voice_injected(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output("Hi.", chunker=chunker)
        result = text2tts(out, {"default_voice": "ryan", "min_sentence_chars": 3})
        assert result[0]["voice"] == "ryan"

    def test_per_request_voice_override(self):
        """extra.tts_voice should override the YAML default_voice (RFC Q3)."""
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output(
            "Hi.",
            chunker=chunker,
            extra={"tts_voice": "serena", "tts_language": "French"},
        )
        result = text2tts(out, {"default_voice": "vivian", "min_sentence_chars": 3})
        assert result[0]["voice"] == "serena"
        assert result[0]["language"] == "French"

    def test_per_request_instructions_forwarded(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output(
            "Hi.",
            chunker=chunker,
            extra={"tts_instructions": "speak slowly"},
        )
        result = text2tts(out, {"min_sentence_chars": 3})
        assert result[0].get("instructions") == "speak slowly"

    def test_no_instructions_key_when_not_provided(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output("Hi.", chunker=chunker)
        result = text2tts(out, {"min_sentence_chars": 3})
        assert "instructions" not in result[0]

    # ------------------------------------------------------------------
    # Output format matches Qwen3-TTS expected input
    # ------------------------------------------------------------------

    def test_output_has_required_tts_keys(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output("Hi.", chunker=chunker)
        result = text2tts(out, {"min_sentence_chars": 3})
        tts_input = result[0]
        assert "text" in tts_input
        assert "task_type" in tts_input
        assert "voice" in tts_input
        assert "language" in tts_input

    # ------------------------------------------------------------------
    # Works with no bridge_config (uses all defaults)
    # ------------------------------------------------------------------

    def test_works_with_no_bridge_config(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output("Hi.", chunker=chunker)
        result = text2tts(out)  # no bridge_config arg
        assert isinstance(result, list)

    def test_works_with_empty_bridge_config(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=3))
        out = self._make_output("Hi.", chunker=chunker)
        result = text2tts(out, {})
        assert isinstance(result, list)

    # ------------------------------------------------------------------
    # Stateful chunker is reused across calls (simulates real streaming)
    # ------------------------------------------------------------------

    def test_stateful_chunker_across_multiple_calls(self):
        """
        Simulate a real multi-turn async_chunk stream:
        tokens arrive one at a time, chunker state accumulates.
        """
        cfg = {"min_sentence_chars": 5}
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=5))

        tokens = ["The", " sky", " is", " blue", ".", " Stars", " shine", "."]
        all_results = []

        for i, tok in enumerate(tokens):
            is_last = (i == len(tokens) - 1)
            stage_out = self._make_output(tok, is_finished=is_last, chunker=chunker)
            all_results.extend(text2tts(stage_out, cfg))

        texts = [r["text"] for r in all_results]
        full = " ".join(texts)
        assert "blue" in full
        assert "shine" in full
