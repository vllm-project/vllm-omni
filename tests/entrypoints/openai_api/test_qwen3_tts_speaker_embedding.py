# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for speaker_embedding API parameter.

Tests Base models (0.6B-Base and 1.7B-Base) which support voice cloning
via pre-computed ECAPA-TDNN embeddings, bypassing ref_audio extraction.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import struct

import httpx
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_BASE_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# A synthetic 1024-dim speaker embedding (all 0.1 — not a real voice, but
# exercises the full code path through the talker's _build_prompt_embeds).
DUMMY_EMBEDDING_1024 = [0.1] * 1024
DUMMY_EMBEDDING_2048 = [0.1] * 2048

SYN_TEXT = "Hello."
MIN_AUDIO_BYTES = 2000
# Limit generation to keep tests fast (dummy embeddings produce nonsensical
# output that may never hit a natural stop token).
MAX_NEW_TOKENS = 256


base_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL_BASE,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--enforce-eager", "--disable-log-stats"],
        ),
        id="qwen3-tts-0.6b-base",
    )
]

base_1_7b_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL_BASE_1_7B,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--enforce-eager", "--disable-log-stats"],
        ),
        id="qwen3-tts-1.7b-base",
    )
]


def verify_wav_audio(content: bytes) -> bool:
    if len(content) < 44:
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


def assert_not_silence(pcm_bytes: bytes):
    """Assert PCM16 samples are not all identical (e.g. all-silence)."""
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    unique = set(samples)
    assert len(unique) > 1, f"All-silence detected: {len(samples)} samples, unique values: {unique}"


# ── 0.6B-Base model tests ──


@pytest.mark.parametrize("omni_server", base_server_params, indirect=True)
class TestSpeakerEmbeddingBase:
    """Speaker embedding tests against the 0.6B-Base model (supports Base task)."""

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_produces_audio(self, omni_server) -> None:
        """speaker_embedding with Base task produces valid WAV audio."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "wav",
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV"
        assert len(response.content) > MIN_AUDIO_BYTES, f"Audio too small: {len(response.content)} bytes"

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_pcm_not_silence(self, omni_server) -> None:
        """speaker_embedding PCM output contains real audio, not all-silence."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "pcm",
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert len(response.content) > MIN_AUDIO_BYTES
        assert_not_silence(response.content)

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_streaming(self, omni_server) -> None:
        """speaker_embedding works with streaming PCM output."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "pcm",
            "stream": True,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert "audio/pcm" in response.headers.get("content-type", "")
        assert len(response.content) > MIN_AUDIO_BYTES
        assert_not_silence(response.content)

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_mutually_exclusive_with_ref_audio(self, omni_server) -> None:
        """Sending both speaker_embedding and ref_audio returns 400."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "ref_audio": "https://example.com/audio.wav",
            "response_format": "wav",
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
        assert "mutually exclusive" in response.text

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_empty_rejected(self, omni_server) -> None:
        """Empty speaker_embedding list returns 400."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": [],
            "response_format": "wav",
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
        assert "non-empty" in response.text


# ── 1.7B-Base model tests (2048-dim embeddings) ──


@pytest.mark.parametrize("omni_server", base_1_7b_server_params, indirect=True)
class TestSpeakerEmbedding1_7B:
    """Speaker embedding tests against the 1.7B-Base model (2048-dim embeddings)."""

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_2048_dim_embedding_produces_audio(self, omni_server) -> None:
        """2048-dim speaker_embedding with 1.7B-Base model produces valid WAV audio."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_2048,
            "x_vector_only_mode": True,
            "response_format": "wav",
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV"
        assert len(response.content) > MIN_AUDIO_BYTES, f"Audio too small: {len(response.content)} bytes"

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_1024_dim_on_1_7b_model_rejected(self, omni_server) -> None:
        """1024-dim speaker_embedding on a 1.7B model (expects 2048) should fail gracefully."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "wav",
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        # Wrong dimensions should produce an error, not silent garbage.
        assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
        assert "speaker_embedding has 1024 dimensions" in response.text
        assert "expected 2048" in response.text
