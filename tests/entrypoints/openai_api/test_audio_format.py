# SPDX-License-Identifier: Apache-2.0
"""Tests for audio output format handling in chat completions.

Covers:
- #4716: audio.format parameter must be respected, not hardcoded to WAV
- Format validation rejects unsupported formats
- pcm16 is mapped to pcm for soundfile compatibility
- Default format is WAV when not specified
- create_audio encodes correctly for each supported format
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import (
    DEFAULT_AUDIO_FORMAT,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_CHAT_AUDIO_FORMATS,
    CreateAudio,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestAudioFormatConstants:
    def test_default_format_is_wav(self):
        assert DEFAULT_AUDIO_FORMAT == "wav"

    def test_supported_formats_no_aac(self):
        assert "aac" not in SUPPORTED_AUDIO_FORMATS

    def test_chat_formats_include_pcm16(self):
        assert "pcm16" in SUPPORTED_CHAT_AUDIO_FORMATS

    def test_chat_formats_superset_of_audio_formats(self):
        assert SUPPORTED_AUDIO_FORMATS <= SUPPORTED_CHAT_AUDIO_FORMATS


class TestCreateAudio:
    @pytest.fixture
    def mixin(self):
        return AudioMixin()

    @pytest.fixture
    def audio_tensor(self):
        return np.sin(np.linspace(0, 2 * np.pi, 24000)).astype(np.float32)

    @pytest.mark.parametrize("fmt", ["wav", "mp3", "flac", "opus", "pcm"])
    def test_create_audio_supported_formats(self, mixin, audio_tensor, fmt):
        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=24000,
            response_format=fmt,
            speed=1.0,
            base64_encode=False,
        )
        response = mixin.create_audio(audio_obj)
        assert len(response.audio_data) > 0

    def test_wav_magic_bytes(self, mixin, audio_tensor):
        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            base64_encode=False,
        )
        response = mixin.create_audio(audio_obj)
        assert response.audio_data[:4] == b"RIFF"
        assert response.media_type == "audio/wav"

    def test_mp3_encoding(self, mixin, audio_tensor):
        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=24000,
            response_format="mp3",
            speed=1.0,
            base64_encode=False,
        )
        response = mixin.create_audio(audio_obj)
        assert response.media_type == "audio/mpeg"
        assert response.audio_data[:4] != b"RIFF"

    def test_flac_magic_bytes(self, mixin, audio_tensor):
        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=24000,
            response_format="flac",
            speed=1.0,
            base64_encode=False,
        )
        response = mixin.create_audio(audio_obj)
        assert response.audio_data[:4] == b"fLaC"
        assert response.media_type == "audio/flac"

    def test_unsupported_format_falls_back_to_default(self, mixin, audio_tensor):
        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=24000,
            response_format="aac",
            speed=1.0,
            base64_encode=False,
        )
        response = mixin.create_audio(audio_obj)
        assert response.audio_data[:4] == b"RIFF"
        assert response.media_type == "audio/wav"

    def test_base64_encoding(self, mixin, audio_tensor):
        import base64

        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            base64_encode=True,
        )
        response = mixin.create_audio(audio_obj)
        decoded = base64.b64decode(response.audio_data)
        assert decoded[:4] == b"RIFF"


class TestResolveAudioFormat:
    """Test _resolve_audio_format via the serving chat class."""

    @pytest.fixture
    def serving_chat(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        return object.__new__(OmniOpenAIServingChat)

    def _make_request(self, audio_params=None):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        if audio_params is not None:
            req.audio = audio_params
        return req

    def test_default_format_when_no_audio_params(self, serving_chat):
        request = self._make_request()
        result = serving_chat._resolve_audio_format(request)
        assert result == "wav"

    def test_extracts_mp3_format(self, serving_chat):
        request = self._make_request({"format": "mp3", "voice": "alloy"})
        result = serving_chat._resolve_audio_format(request)
        assert result == "mp3"

    def test_pcm16_mapped_to_pcm(self, serving_chat):
        request = self._make_request({"format": "pcm16", "voice": "alloy"})
        result = serving_chat._resolve_audio_format(request)
        assert result == "pcm"

    def test_invalid_format_returns_error(self, serving_chat):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse

        request = self._make_request({"format": "aac", "voice": "alloy"})
        result = serving_chat._resolve_audio_format(request)
        assert isinstance(result, ErrorResponse)
        assert "aac" in result.error.message

    def test_all_supported_formats_accepted(self, serving_chat):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse

        for fmt in SUPPORTED_CHAT_AUDIO_FORMATS:
            request = self._make_request({"format": fmt, "voice": "alloy"})
            result = serving_chat._resolve_audio_format(request)
            assert not isinstance(result, ErrorResponse), f"Format {fmt} should be accepted"
