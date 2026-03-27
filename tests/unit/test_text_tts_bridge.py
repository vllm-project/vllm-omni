# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
tests/unit/test_text_tts_bridge.py
===================================
Unit tests for the text→TTS bridge processor.

GPU-free, model-free. Mocks transfer_manager and request to match the
real OmniChunkTransferAdapter / OmniEngineCoreRequest interfaces.

Run:
    pytest tests/unit/test_text_tts_bridge.py -v --noconftest
"""

from unittest.mock import MagicMock  # noqa: F401

import pytest
from pytest_mock import MockerFixture  # noqa: F401

from vllm_omni.model_executor.stage_input_processors.text_tts_bridge import (
    SentenceChunker,
    TextTTSBridgeConfig,
    _cleanup_chunker,
    _get_decoded_text,
    _get_or_create_chunker,
    text2tts,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_transfer_manager(bridge_cfg: dict | None = None):
    """
    Mock OmniChunkTransferAdapter with the fields our bridge uses:
      - request_payload  (dict, keyed by request_id)
      - connector.config (dict with optional 'bridge' sub-dict)
    """
    tm = MagicMock()
    tm.request_payload = {}
    connector = MagicMock()
    connector.config = {"bridge": bridge_cfg or {}}
    tm.connector = connector
    return tm


def make_request(
    request_id: str = "req-001",
    speaker: str | None = None,
    language: str | None = None,
    output_text: str = "",
):
    """
    Mock OmniEngineCoreRequest with fields our bridge reads:
      - external_req_id
      - additional_information.entries  (for speaker/language)
      - output_text                     (fallback text extraction)
    """
    req = MagicMock()
    req.external_req_id = request_id
    req.output_text = output_text

    # Build additional_information structure matching tts_utils expectations
    entries = {}
    if speaker:
        speaker_entry = MagicMock()
        speaker_entry.list_data = [speaker]
        entries["speaker"] = speaker_entry
    if language:
        lang_entry = MagicMock()
        lang_entry.list_data = [language]
        entries["language"] = lang_entry

    req.additional_information = MagicMock()
    req.additional_information.entries = entries
    return req


def make_pooling_output(text: str = "") -> dict:
    """Minimal pooling_output dict with detokenized text."""
    return {"text": text}


# =============================================================================
# TextTTSBridgeConfig
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
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
        assert cfg.default_language == "English"

    def test_from_dict_ignores_unknown_keys(self):
        cfg = TextTTSBridgeConfig.from_dict({"unknown_key": "value", "default_voice": "ryan"})
        assert cfg.default_voice == "ryan"

    def test_from_dict_empty(self):
        cfg = TextTTSBridgeConfig.from_dict({})
        assert cfg == TextTTSBridgeConfig()


# =============================================================================
# SentenceChunker
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestSentenceChunker:
    def _chunker(self, min_chars=5):
        return SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=min_chars))

    def test_no_flush_without_delimiter(self):
        chunker = self._chunker()
        assert chunker.feed("Hello world") == []

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
        chunker = self._chunker(min_chars=3)
        tokens = list("Hello. World.")
        all_chunks = []
        for tok in tokens:
            all_chunks.extend(chunker.feed(tok))
        all_chunks.extend(chunker.flush())
        full = " ".join(all_chunks)
        assert "Hello." in full
        assert "World." in full

    def test_flush_returns_remaining_buffer(self):
        chunker = self._chunker(min_chars=100)
        chunker.feed("This has no ending punctuation")
        result = chunker.flush()
        assert len(result) == 1
        assert "no ending" in result[0]

    def test_flush_empty_buffer_returns_empty(self):
        assert self._chunker().flush() == []

    def test_flush_whitespace_only_returns_empty(self):
        chunker = self._chunker()
        chunker.feed("   ")
        assert chunker.flush() == []

    def test_double_flush_is_safe(self):
        chunker = self._chunker()
        chunker.feed("Some text")
        chunker.flush()
        assert chunker.flush() == []

    def test_min_chars_prevents_premature_flush(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=50))
        assert chunker.feed("Hi.") == []
        assert chunker.flush() == ["Hi."]

    def test_min_chars_zero_flushes_immediately(self):
        chunker = SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=0))
        assert chunker.feed("Hi.") == ["Hi."]

    def test_chinese_delimiter(self):
        chunker = self._chunker(min_chars=2)
        result = chunker.feed("你好。")
        assert len(result) == 1

    def test_custom_delimiters(self):
        cfg = TextTTSBridgeConfig(sentence_delimiters=["|"], min_sentence_chars=2)
        chunker = SentenceChunker(cfg)
        result = chunker.feed("chunk one| chunk two|")
        assert len(result) == 2


# =============================================================================
# _get_decoded_text
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestGetDecodedText:
    def test_reads_text_key_from_pooling_output(self):
        req = make_request()
        text = _get_decoded_text({"text": "Hello"}, req)
        assert text == "Hello"

    def test_reads_list_text_key(self):
        req = make_request()
        text = _get_decoded_text({"text": ["tok1", "tok2"]}, req)
        assert text == "tok2"

    def test_falls_back_to_request_output_text(self):
        req = make_request(output_text="fallback text")
        text = _get_decoded_text({}, req)
        assert text == "fallback text"

    def test_returns_empty_string_when_nothing(self):
        req = make_request()
        text = _get_decoded_text({}, req)
        assert text == ""


# =============================================================================
# transfer_manager state helpers
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestTransferManagerState:
    def test_creates_chunker_on_first_call(self):
        tm = make_transfer_manager()
        cfg = TextTTSBridgeConfig(min_sentence_chars=5)
        chunker = _get_or_create_chunker(tm, "req-1", cfg)
        assert isinstance(chunker, SentenceChunker)

    def test_returns_same_chunker_on_second_call(self):
        tm = make_transfer_manager()
        cfg = TextTTSBridgeConfig(min_sentence_chars=5)
        c1 = _get_or_create_chunker(tm, "req-1", cfg)
        c2 = _get_or_create_chunker(tm, "req-1", cfg)
        assert c1 is c2

    def test_different_request_ids_get_different_chunkers(self):
        tm = make_transfer_manager()
        cfg = TextTTSBridgeConfig()
        c1 = _get_or_create_chunker(tm, "req-1", cfg)
        c2 = _get_or_create_chunker(tm, "req-2", cfg)
        assert c1 is not c2

    def test_cleanup_removes_chunker(self):
        tm = make_transfer_manager()
        cfg = TextTTSBridgeConfig()
        _get_or_create_chunker(tm, "req-1", cfg)
        _cleanup_chunker(tm, "req-1")
        key = "_tts_chunker_req-1"
        assert key not in tm.request_payload

    def test_cleanup_is_safe_when_key_missing(self):
        tm = make_transfer_manager()
        _cleanup_chunker(tm, "nonexistent")  # should not raise


# =============================================================================
# text2tts — the full hook with real signature
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestText2TtsHook:
    """
    Tests use the real function signature matching OmniChunkTransferAdapter:
        text2tts(transfer_manager, pooling_output, request, is_finished)
    """

    # ------------------------------------------------------------------
    # Buffering behavior
    # ------------------------------------------------------------------

    def test_returns_none_when_buffering(self):
        tm = make_transfer_manager({"min_sentence_chars": 5})
        req = make_request()
        result = text2tts(tm, make_pooling_output("Hello world"), req, False)
        assert result is None

    def test_returns_chunks_when_sentence_complete(self):
        tm = make_transfer_manager({"min_sentence_chars": 3})
        req = make_request()
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result is not None
        assert isinstance(result, dict)
        assert result["text"] == "Hi."

    def test_flushes_on_eos(self):
        tm = make_transfer_manager({"min_sentence_chars": 100})
        req = make_request()
        # Feed text without delimiter
        text2tts(tm, make_pooling_output("No delimiter here"), req, False)
        # Send EOS
        result = text2tts(tm, make_pooling_output(""), req, True)
        assert result is not None
        assert isinstance(result, dict)
        assert "No delimiter here" in result["text"]

    def test_cleanup_after_eos(self):
        tm = make_transfer_manager({"min_sentence_chars": 3})
        req = make_request()
        text2tts(tm, make_pooling_output("Hi."), req, True)
        key = f"_tts_chunker_{req.external_req_id}"
        assert key not in tm.request_payload

    # ------------------------------------------------------------------
    # Output format matches Qwen3-TTS expected input
    # ------------------------------------------------------------------

    def test_output_has_required_keys(self):
        tm = make_transfer_manager({"min_sentence_chars": 3})
        req = make_request()
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result is not None
        assert "text" in result
        assert "task_type" in result
        assert "voice" in result
        assert "language" in result

    # ------------------------------------------------------------------
    # Voice/language injection — RFC Q3
    # ------------------------------------------------------------------

    def test_default_voice_from_yaml_config(self):
        tm = make_transfer_manager({"min_sentence_chars": 3, "default_voice": "ryan"})
        req = make_request()  # no speaker in request
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result["voice"] == "ryan"

    def test_per_request_voice_overrides_default(self):
        """Speaker from request.additional_information overrides YAML default."""
        tm = make_transfer_manager({"min_sentence_chars": 3, "default_voice": "vivian"})
        req = make_request(speaker="serena")
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result["voice"] == "serena"

    def test_per_request_language_overrides_default(self):
        tm = make_transfer_manager({"min_sentence_chars": 3, "default_language": "English"})
        req = make_request(language="French")
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result["language"] == "French"

    # ------------------------------------------------------------------
    # Stateful chunker persists across multiple calls (real streaming)
    # ------------------------------------------------------------------

    def test_stateful_across_multiple_calls(self):
        """
        Simulate real token-by-token async_chunk streaming.
        Chunker state must accumulate correctly across calls.
        """
        tm = make_transfer_manager({"min_sentence_chars": 5})
        req = make_request()

        tokens = ["The", " sky", " is", " blue", ".", " Stars", " shine", "."]
        all_results = []

        for i, tok in enumerate(tokens):
            is_last = i == len(tokens) - 1
            r = text2tts(tm, make_pooling_output(tok), req, is_last)
            if r:
                all_results.append(r)

        texts = " ".join(r["text"] for r in all_results)
        assert "blue" in texts
        assert "shine" in texts

    def test_multiple_requests_isolated(self):
        """Two concurrent requests must not share chunker state."""
        tm = make_transfer_manager({"min_sentence_chars": 3})
        req1 = make_request(request_id="req-A")
        req2 = make_request(request_id="req-B")

        text2tts(tm, make_pooling_output("Hello"), req1, False)
        text2tts(tm, make_pooling_output("World"), req2, False)

        key_a = "_tts_chunker_req-A"
        key_b = "_tts_chunker_req-B"
        assert key_a in tm.request_payload
        assert key_b in tm.request_payload
        assert tm.request_payload[key_a] is not tm.request_payload[key_b]
