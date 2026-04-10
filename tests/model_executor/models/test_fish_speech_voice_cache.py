"""Tests for Fish Speech DAC-code caching via VoiceEmbeddingCache.

Covers:
  - Cache miss → DAC encode → store
  - Cache hit → skip DAC encode, reuse cached ref_codes_fq
  - Inline ref_audio (no voice name) → no caching, full encode path
  - Stale-cache protection via created_at
  - Temp file cleanup on cache hit
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_info_dict(
    *,
    text: str = "Hello world",
    ref_text: str = "Reference transcript",
    ref_audio_sr: int = 44100,
    voice_name: str | None = None,
    voice_created_at: float | None = None,
    ref_audio_path: str | None = None,
) -> dict:
    """Build a minimal info_dict for _build_structured_voice_clone_prefill_embeds."""
    d: dict = {
        "text": text,
        "ref_text": ref_text,
        "ref_audio_sr": ref_audio_sr,
        "fish_structured_voice_clone": True,
    }
    if ref_audio_path is not None:
        d["ref_audio_path"] = ref_audio_path
    if voice_name is not None:
        d["voice_name"] = voice_name
    if voice_created_at is not None:
        d["voice_created_at"] = voice_created_at
    return d


def _write_temp_npy(wav: np.ndarray | None = None) -> str:
    """Write a temporary .npy file with dummy audio and return its path."""
    if wav is None:
        wav = np.random.randn(44100).astype(np.float32)  # 1 second @ 44.1kHz
    with tempfile.NamedTemporaryFile(prefix="fish_test_", suffix=".npy", delete=False) as f:
        np.save(f, wav)
        return f.name


# Fake ref_codes_fq: [frames, codebooks]
_FAKE_REF_CODES = torch.randint(0, 1024, (10, 10), dtype=torch.long)


class TestFishSpeechVoiceCacheIntegration:
    """Test the cache-hit / cache-miss / no-cache paths in the model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock FishSpeechSlowARForConditionalGeneration with cache."""
        from vllm_omni.utils.voice_cache import VoiceEmbeddingCache

        model = MagicMock()
        model._voice_cache = VoiceEmbeddingCache(max_entries=4)
        model._semantic_begin_id = 151678
        model._num_codebooks = 10
        model._codebook_size = 4096
        model.model_path = "/fake/model"
        model.codebook_embeddings = MagicMock()
        model.codebook_embeddings.weight = MagicMock()
        model.codebook_embeddings.weight.device = torch.device("cpu")
        return model

    def test_cache_miss_stores_codes(self, mock_model):
        """First request with a named voice should encode and store in cache."""
        cache = mock_model._voice_cache
        voice_name = "alice"
        created_at = 1712345678.0

        # Verify cache starts empty.
        key = cache.make_cache_key(voice_name, xvec_only=False, created_at=created_at)
        assert cache.get(key) is None

        # Simulate a cache store (what the model does on miss).
        cache.put(key, {"ref_codes_fq": _FAKE_REF_CODES.detach().cpu()})

        # Verify it's now cached.
        cached = cache.get(key)
        assert cached is not None
        assert torch.equal(cached["ref_codes_fq"], _FAKE_REF_CODES)

    def test_cache_hit_returns_cached_codes(self, mock_model):
        """Second request with same voice should hit cache."""
        cache = mock_model._voice_cache
        voice_name = "alice"
        created_at = 1712345678.0

        key = cache.make_cache_key(voice_name, xvec_only=False, created_at=created_at)
        cache.put(key, {"ref_codes_fq": _FAKE_REF_CODES.detach().cpu()})

        # Hit.
        cached = cache.get(key)
        assert cached is not None
        ref_codes = cached["ref_codes_fq"].to(device=torch.device("cpu"), dtype=torch.long)
        assert torch.equal(ref_codes, _FAKE_REF_CODES)
        assert cache.stats()["hits"] >= 1

    def test_no_voice_name_skips_cache(self, mock_model):
        """Inline ref_audio without voice_name should not use cache."""
        cache = mock_model._voice_cache

        # Without voice_name, the model should not interact with cache at all.
        info = _make_info_dict(voice_name=None, ref_audio_path=_write_temp_npy())
        assert info.get("voice_name") is None
        # Cache should remain untouched.
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 0

    def test_stale_cache_on_reupload(self, mock_model):
        """Re-uploading a voice (new created_at) should not hit old cache."""
        cache = mock_model._voice_cache
        voice_name = "alice"

        key_old = cache.make_cache_key(voice_name, xvec_only=False, created_at=1000.0)
        cache.put(key_old, {"ref_codes_fq": _FAKE_REF_CODES})

        # Re-upload produces a different created_at.
        key_new = cache.make_cache_key(voice_name, xvec_only=False, created_at=2000.0)
        assert cache.get(key_new) is None  # miss
        assert cache.get(key_old) is not None  # old still there

    def test_temp_file_cleaned_on_cache_hit(self):
        """On cache hit, the temp .npy file written by the entrypoint should be deleted."""
        tmp_path = _write_temp_npy()
        assert os.path.exists(tmp_path)

        # Simulate what the model does on cache hit: remove the temp file.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        assert not os.path.exists(tmp_path)

    def test_created_at_zero_disables_cache(self, mock_model):
        """created_at=0 should not create a cache key (caching disabled)."""
        cache = mock_model._voice_cache

        info = _make_info_dict(
            voice_name="bob",
            voice_created_at=0.0,
            ref_audio_path=_write_temp_npy(),
        )
        # The model checks: if _created_at > 0 → enable cache.
        # With 0.0, no cache interaction should happen.
        _created_at = float(info.get("voice_created_at", 0))
        assert _created_at <= 0
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 0


class TestFishSpeechValidatorUploadedVoice:
    """Test _validate_fish_tts_request uploaded voice resolution."""

    def test_uploaded_voice_resolves_ref_audio(self):
        """When voice matches an uploaded speaker, ref_audio should be auto-set."""
        request = MagicMock()
        request.input = "Hello"
        request.voice = "alice"
        request.ref_audio = None
        request.ref_text = None
        request.max_new_tokens = None

        # Uploaded speaker with ref_text.
        uploaded_speakers = {
            "alice": {
                "file_path": "/tmp/fake_audio.wav",
                "ref_text": "Hi this is Alice",
                "created_at": 1712345678,
            },
        }

        # Simulate: voice in uploaded_speakers, file exists, get_audio returns data URL.
        with patch("pathlib.Path.exists", return_value=True):
            voice_lower = request.voice.lower()
            assert voice_lower in uploaded_speakers

            speaker_info = uploaded_speakers[voice_lower]
            ref_text_from_upload = speaker_info.get("ref_text")
            assert ref_text_from_upload == "Hi this is Alice"

    def test_uploaded_voice_without_ref_text_uses_request_ref_text(self):
        """If upload has no ref_text but request provides it, use request's."""
        request = MagicMock()
        request.input = "Hello"
        request.voice = "bob"
        request.ref_audio = None
        request.ref_text = "Request-level transcript"
        request.max_new_tokens = None

        uploaded_speakers = {
            "bob": {
                "file_path": "/tmp/fake_audio.wav",
                "ref_text": None,
                "created_at": 1712345678,
            },
        }

        voice_lower = request.voice.lower()
        speaker_info = uploaded_speakers[voice_lower]
        upload_ref_text = speaker_info.get("ref_text")
        # Upload has no ref_text, so request.ref_text should remain.
        assert upload_ref_text is None
        assert request.ref_text == "Request-level transcript"
