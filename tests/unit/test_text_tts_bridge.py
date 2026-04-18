# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
tests/unit/test_text_tts_bridge.py
===================================
Unit tests for the text->TTS bridge processor.

GPU-free, model-free. Uses pytest-mock (mocker fixture) matching
the project convention in tests/benchmarks/patch/test_patch.py.

Run:
    pytest tests/unit/test_text_tts_bridge.py -v --noconftest
"""

import pytest
from pytest_mock import MockerFixture

from vllm_omni.model_executor.stage_input_processors.text_tts_bridge import (
    SentenceChunker,
    TextTTSBridgeConfig,
    _cleanup_chunker,
    _get_decoded_text,
    _get_or_create_chunker,
    text2tts,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_transfer_manager(mocker: MockerFixture, bridge_cfg: dict | None = None):
    """Mock OmniChunkTransferAdapter."""
    tm = mocker.MagicMock()
    tm.request_payload = {}
    connector = mocker.MagicMock()
    connector.config = {"bridge": bridge_cfg or {}}
    tm.connector = connector
    return tm


def make_request(
    mocker: MockerFixture,
    request_id: str = "req-001",
    speaker: str | None = None,
    language: str | None = None,
    output_text: str = "",
):
    """Mock OmniEngineCoreRequest."""
    req = mocker.MagicMock()
    req.external_req_id = request_id
    req.output_text = output_text

    entries = {}
    if speaker:
        speaker_entry = mocker.MagicMock()
        speaker_entry.list_data = [speaker]
        entries["speaker"] = speaker_entry
    if language:
        lang_entry = mocker.MagicMock()
        lang_entry.list_data = [language]
        entries["language"] = lang_entry

    req.additional_information = mocker.MagicMock()
    req.additional_information.entries = entries
    return req


def make_pooling_output(text: str = "") -> dict:
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
        assert TextTTSBridgeConfig.from_dict({}) == TextTTSBridgeConfig()


# =============================================================================
# SentenceChunker
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestSentenceChunker:
    def _chunker(self, min_chars=5):
        return SentenceChunker(TextTTSBridgeConfig(min_sentence_chars=min_chars))

    def test_no_flush_without_delimiter(self):
        assert self._chunker().feed("Hello world") == []

    def test_simple_sentence_flush(self):
        result = self._chunker(min_chars=5).feed("Hello!")
        assert len(result) == 1
        assert result[0] == "Hello!"

    def test_multiple_sentences_in_one_feed(self):
        result = self._chunker(min_chars=3).feed("One. Two. Three.")
        assert len(result) == 3
        assert result[0] == "One."
        assert result[1] == "Two."
        assert result[2] == "Three."

    def test_incremental_token_feeding(self):
        chunker = self._chunker(min_chars=3)
        all_chunks = []
        for tok in list("Hello. World."):
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
        result = self._chunker(min_chars=2).feed("你好。")
        assert len(result) == 1

    def test_custom_delimiters(self):
        cfg = TextTTSBridgeConfig(sentence_delimiters=["|"], min_sentence_chars=2)
        result = SentenceChunker(cfg).feed("chunk one| chunk two|")
        assert len(result) == 2


# =============================================================================
# _get_decoded_text
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestGetDecodedText:
    def test_reads_text_key_from_pooling_output(self, mocker: MockerFixture):
        req = make_request(mocker)
        assert _get_decoded_text({"text": "Hello"}, req) == "Hello"

    def test_reads_list_text_key(self, mocker: MockerFixture):
        req = make_request(mocker)
        assert _get_decoded_text({"text": ["tok1", "tok2"]}, req) == "tok2"

    def test_falls_back_to_request_output_text(self, mocker: MockerFixture):
        req = make_request(mocker, output_text="fallback text")
        assert _get_decoded_text({}, req) == "fallback text"

    def test_returns_empty_string_when_nothing(self, mocker: MockerFixture):
        req = make_request(mocker)
        assert _get_decoded_text({}, req) == ""


# =============================================================================
# transfer_manager state helpers
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestTransferManagerState:
    def test_creates_chunker_on_first_call(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker)
        chunker = _get_or_create_chunker(tm, "req-1", TextTTSBridgeConfig())
        assert isinstance(chunker, SentenceChunker)

    def test_returns_same_chunker_on_second_call(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker)
        cfg = TextTTSBridgeConfig()
        c1 = _get_or_create_chunker(tm, "req-1", cfg)
        c2 = _get_or_create_chunker(tm, "req-1", cfg)
        assert c1 is c2

    def test_different_request_ids_get_different_chunkers(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker)
        cfg = TextTTSBridgeConfig()
        assert _get_or_create_chunker(tm, "req-1", cfg) is not _get_or_create_chunker(tm, "req-2", cfg)

    def test_cleanup_removes_chunker(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker)
        _get_or_create_chunker(tm, "req-1", TextTTSBridgeConfig())
        _cleanup_chunker(tm, "req-1")
        assert "_tts_chunker_req-1" not in tm.request_payload

    def test_cleanup_is_safe_when_key_missing(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker)
        _cleanup_chunker(tm, "nonexistent")  # should not raise


# =============================================================================
# text2tts hook
# =============================================================================


@pytest.mark.core_model
@pytest.mark.cpu
class TestText2TtsHook:
    def test_returns_none_when_buffering(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 5})
        req = make_request(mocker)
        assert text2tts(tm, make_pooling_output("Hello world"), req, False) is None

    def test_returns_chunk_when_sentence_complete(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 3})
        req = make_request(mocker)
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result is not None
        assert isinstance(result, dict)
        assert result["text"] == "Hi."

    def test_flushes_on_eos(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 100})
        req = make_request(mocker)
        text2tts(tm, make_pooling_output("No delimiter here"), req, False)
        result = text2tts(tm, make_pooling_output(""), req, True)
        assert result is not None
        assert isinstance(result, dict)
        assert "No delimiter here" in result["text"]  # all chunks combined

    def test_cleanup_after_eos(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 3})
        req = make_request(mocker)
        text2tts(tm, make_pooling_output("Hi."), req, True)
        assert f"_tts_chunker_{req.external_req_id}" not in tm.request_payload

    def test_output_has_required_keys(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 3})
        req = make_request(mocker)
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result is not None
        for key in ("text", "task_type", "voice", "language"):
            assert key in result

    def test_default_voice_from_yaml_config(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 3, "default_voice": "ryan"})
        req = make_request(mocker)
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result["voice"] == "ryan"

    def test_per_request_voice_overrides_default(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 3, "default_voice": "vivian"})
        req = make_request(mocker, speaker="serena")
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result["voice"] == "serena"

    def test_per_request_language_overrides_default(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 3, "default_language": "English"})
        req = make_request(mocker, language="French")
        result = text2tts(tm, make_pooling_output("Hi."), req, False)
        assert result["language"] == "French"

    def test_stateful_across_multiple_calls(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 5})
        req = make_request(mocker)
        tokens = ["The", " sky", " is", " blue", ".", " Stars", " shine", "."]
        all_results = []
        for i, tok in enumerate(tokens):
            r = text2tts(tm, make_pooling_output(tok), req, i == len(tokens) - 1)
            if r:
                all_results.append(r)
        texts = " ".join(r["text"] for r in all_results)
        assert "blue" in texts
        assert "shine" in texts

    def test_multiple_requests_isolated(self, mocker: MockerFixture):
        tm = make_transfer_manager(mocker, {"min_sentence_chars": 3})
        req1 = make_request(mocker, request_id="req-A")
        req2 = make_request(mocker, request_id="req-B")
        text2tts(tm, make_pooling_output("Hello"), req1, False)
        text2tts(tm, make_pooling_output("World"), req2, False)
        assert "_tts_chunker_req-A" in tm.request_payload
        assert "_tts_chunker_req-B" in tm.request_payload
        assert tm.request_payload["_tts_chunker_req-A"] is not tm.request_payload["_tts_chunker_req-B"]
