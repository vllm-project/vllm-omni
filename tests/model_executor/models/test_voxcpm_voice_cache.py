# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for VoxCPM prompt-cache caching via VoiceEmbeddingCache.

Covers:
  - Cache miss → build_prompt_cache → store
  - Cache hit → skip build_prompt_cache, reuse cached artifacts
  - Inline ref_audio (no voice name) → no caching
  - Stale-cache protection via created_at
  - created_at=0 disables caching
"""

import os
import tempfile

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.voxcpm.voxcpm_stage_wrappers import (
    _DirectVoxCPMLatentGenerator,
)
from vllm_omni.utils.voice_cache import VoiceEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FAKE_AUDIO_FEAT = torch.randn(5, 2, 64)
_FAKE_PROMPT_CACHE = {"prompt_text": "hello", "audio_feat": _FAKE_AUDIO_FEAT}


def _write_temp_wav() -> str:
    """Write a trivial temp file to act as a prompt_wav_path."""
    with tempfile.NamedTemporaryFile(prefix="voxcpm_test_", suffix=".wav", delete=False) as f:
        f.write(b"\x00" * 44)
        return f.name


@pytest.fixture
def mock_generator(mocker: MockerFixture) -> _DirectVoxCPMLatentGenerator:
    """Create a _DirectVoxCPMLatentGenerator with a mocked tts_model."""
    tts_model = mocker.MagicMock()
    tts_model.sample_rate = 24000
    tts_model.build_prompt_cache.return_value = {
        "prompt_text": "hello",
        "audio_feat": _FAKE_AUDIO_FEAT.clone(),
    }
    gen = _DirectVoxCPMLatentGenerator(tts_model)
    gen._voice_cache = VoiceEmbeddingCache(max_entries=4)
    return gen


class TestVoxCPMVoiceCacheIntegration:
    def test_cache_miss_stores_prompt_cache(self, mock_generator):
        """First request with a named voice should encode and store."""
        wav_path = _write_temp_wav()
        try:
            result = mock_generator._resolve_prompt_cache(
                wav_path, "hello", voice_name="alice", voice_created_at=1000.0,
            )
            assert result is not None
            assert "audio_feat" in result
            assert "prompt_text" in result

            mock_generator.tts_model.build_prompt_cache.assert_called_once()

            key = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False, created_at=1000.0)
            cached = mock_generator._voice_cache.get(key)
            assert cached is not None
            assert torch.equal(cached["audio_feat"], _FAKE_AUDIO_FEAT)
        finally:
            os.unlink(wav_path)

    def test_cache_hit_skips_build(self, mock_generator):
        """Second request with same voice should hit cache and skip build_prompt_cache."""
        wav_path = _write_temp_wav()
        try:
            mock_generator._resolve_prompt_cache(
                wav_path, "hello", voice_name="alice", voice_created_at=1000.0,
            )
            mock_generator.tts_model.build_prompt_cache.reset_mock()

            result = mock_generator._resolve_prompt_cache(
                wav_path, "hello", voice_name="alice", voice_created_at=1000.0,
            )
            assert result is not None
            mock_generator.tts_model.build_prompt_cache.assert_not_called()
            assert mock_generator._voice_cache.stats()["hits"] >= 1
        finally:
            os.unlink(wav_path)

    def test_no_voice_name_skips_cache(self, mock_generator):
        """Inline ref_audio without voice_name should not use cache."""
        wav_path = _write_temp_wav()
        try:
            result = mock_generator._resolve_prompt_cache(wav_path, "hello")
            assert result is not None

            mock_generator.tts_model.build_prompt_cache.assert_called_once()
            assert mock_generator._voice_cache.stats()["hits"] == 0
            assert mock_generator._voice_cache.stats()["misses"] == 0
            assert mock_generator._voice_cache.stats()["entries"] == 0
        finally:
            os.unlink(wav_path)

    def test_stale_cache_on_reupload(self, mock_generator):
        """Re-uploading a voice (new created_at) should not hit old cache."""
        wav_path = _write_temp_wav()
        try:
            mock_generator._resolve_prompt_cache(
                wav_path, "hello", voice_name="alice", voice_created_at=1000.0,
            )
            mock_generator.tts_model.build_prompt_cache.reset_mock()

            result = mock_generator._resolve_prompt_cache(
                wav_path, "hello", voice_name="alice", voice_created_at=2000.0,
            )
            assert result is not None
            mock_generator.tts_model.build_prompt_cache.assert_called_once()
        finally:
            os.unlink(wav_path)

    def test_created_at_zero_disables_cache(self, mock_generator):
        """created_at=0 should bypass caching entirely."""
        wav_path = _write_temp_wav()
        try:
            mock_generator._resolve_prompt_cache(
                wav_path, "hello", voice_name="alice", voice_created_at=0.0,
            )
            mock_generator._resolve_prompt_cache(
                wav_path, "hello", voice_name="alice", voice_created_at=0.0,
            )
            assert mock_generator.tts_model.build_prompt_cache.call_count == 2
            assert mock_generator._voice_cache.stats()["entries"] == 0
        finally:
            os.unlink(wav_path)

    def test_no_prompt_inputs_returns_none(self, mock_generator):
        """No prompt_wav_path or prompt_text should return None without caching."""
        assert mock_generator._resolve_prompt_cache(None, None) is None
        assert mock_generator._resolve_prompt_cache(None, "text") is None
        assert mock_generator._resolve_prompt_cache("/path", None) is None
        mock_generator.tts_model.build_prompt_cache.assert_not_called()
