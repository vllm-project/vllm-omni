# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for audio output format handling in chat completions.

Covers:
- #4716: audio.format parameter must be respected, not hardcoded to WAV
- Format validation rejects unsupported formats
- pcm16 is mapped to pcm for soundfile compatibility
- Default format is WAV when not specified
- create_audio encodes correctly for each supported format
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import soundfile
import torch
import torchaudio

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin, StreamingAudioResampler
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

    def test_resamples_wav_to_requested_output_rate(self, mixin, audio_tensor):
        response = mixin.create_audio(
            CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=24000,
                output_sample_rate=8000,
                response_format="wav",
                speed=1.0,
                base64_encode=False,
            )
        )

        with soundfile.SoundFile(BytesIO(response.audio_data)) as audio_file:
            assert audio_file.samplerate == 8000
            assert audio_file.frames == 8000


class TestStreamingAudioResampler:
    @staticmethod
    def _resample(waveform):
        resampler = StreamingAudioResampler(24000, 8000)
        return np.concatenate(
            (
                resampler.process(waveform),
                resampler.process(np.empty(0), final=True),
            )
        )

    def test_chunk_boundaries_do_not_change_output(self):
        waveform = np.sin(np.linspace(0, 200 * np.pi, 24000, endpoint=False)).astype(np.float32)

        expected = self._resample(waveform)

        chunked = StreamingAudioResampler(24000, 8000)
        pieces = [chunked.process(chunk) for chunk in np.split(waveform, [137, 2048, 9001, 17003])]
        pieces.append(chunked.process(np.empty(0), final=True))
        actual = np.concatenate(pieces)

        assert actual.shape == (8000,)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

        reference = torchaudio.functional.resample(torch.from_numpy(waveform), 24000, 8000).numpy()
        np.testing.assert_allclose(actual[100:-100], reference[100:-100], atol=2e-3, rtol=1e-3)

    def test_preserves_passband_signal(self):
        samples = np.arange(24000, dtype=np.float32)
        waveform = np.sin(2 * np.pi * 3000 * samples / 24000).astype(np.float32)

        output = self._resample(waveform)

        input_rms = np.sqrt(np.mean(waveform**2))
        output_rms = np.sqrt(np.mean(output[100:-100] ** 2))
        assert output_rms == pytest.approx(input_rms, rel=0.02)

    @pytest.mark.parametrize("frequency", [6000, 10000])
    def test_attenuates_aliasing_frequencies(self, frequency):
        samples = np.arange(24000, dtype=np.float32)
        waveform = np.sin(2 * np.pi * frequency * samples / 24000).astype(np.float32)

        output = self._resample(waveform)

        input_rms = np.sqrt(np.mean(waveform**2))
        output_rms = np.sqrt(np.mean(output[100:-100] ** 2))
        assert output_rms < input_rms * 0.01

    def test_rejects_non_integer_downsampling_ratio(self):
        with pytest.raises(ValueError, match="integer downsampling ratio"):
            StreamingAudioResampler(24000, 16000)

    def test_rejects_multichannel_audio(self):
        resampler = StreamingAudioResampler(24000, 8000)

        with pytest.raises(ValueError, match="only supports mono audio"):
            resampler.process(np.zeros((2, 240), dtype=np.float32))


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
