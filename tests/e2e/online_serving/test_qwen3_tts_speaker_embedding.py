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
from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path

MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_BASE_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
STAGE_INIT_TIMEOUT_S = 120

# A synthetic 1024-dim speaker embedding (all 0.1 — not a real voice, but
# exercises the full code path through the talker's _build_prompt_embeds).
DUMMY_EMBEDDING_1024 = [0.1] * 1024
DUMMY_EMBEDDING_2048 = [0.1] * 2048

SYN_TEXT = "Hello."
MIN_AUDIO_BYTES = 2000
# Limit generation to keep tests fast (dummy embeddings produce nonsensical
# output that may never hit a natural stop token).
MAX_NEW_TOKENS = 256


def get_stage_config() -> str:
    return get_deploy_config_path("qwen3_tts.yaml")


def _server_args():
    return [
        "--stage-configs-path",
        get_stage_config(),
        "--stage-init-timeout",
        str(STAGE_INIT_TIMEOUT_S),
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-log-stats",
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


@pytest.fixture(scope="module")
def base_server():
    """Start vLLM-Omni server with 0.6B-Base model."""
    with OmniServer(MODEL_BASE, _server_args()) as server:
        yield server


class TestSpeakerEmbeddingBase:
    """Speaker embedding tests against the 0.6B-Base model (supports Base task)."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_produces_audio(self, base_server) -> None:
        """speaker_embedding with Base task produces valid WAV audio."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
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

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_pcm_not_silence(self, base_server) -> None:
        """speaker_embedding PCM output contains real audio, not all-silence."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
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

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_streaming(self, base_server) -> None:
        """speaker_embedding works with streaming PCM output."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
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

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_mutually_exclusive_with_ref_audio(self, base_server) -> None:
        """Sending both speaker_embedding and ref_audio returns 400."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
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

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_empty_rejected(self, base_server) -> None:
        """Empty speaker_embedding list returns 400."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
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


@pytest.fixture(scope="module")
def base_1_7b_server():
    """Start vLLM-Omni server with 1.7B-Base model."""
    with OmniServer(MODEL_BASE_1_7B, _server_args()) as server:
        yield server


class TestSpeakerEmbedding1_7B:
    """Speaker embedding tests against the 1.7B-Base model (2048-dim embeddings)."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_2048_dim_embedding_produces_audio(self, base_1_7b_server) -> None:
        """2048-dim speaker_embedding with 1.7B-Base model produces valid WAV audio."""
        url = f"http://{base_1_7b_server.host}:{base_1_7b_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE_1_7B,
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

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_1024_dim_on_1_7b_model_rejected(self, base_1_7b_server) -> None:
        """1024-dim speaker_embedding on a 1.7B model (expects 2048) should fail gracefully."""
        url = f"http://{base_1_7b_server.host}:{base_1_7b_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE_1_7B,
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
