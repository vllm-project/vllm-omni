# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import httpx
import pytest

from tests.conftest import OmniServer
from tests.utils import hardware_test

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def get_stage_config():
    """Get the stage config path for Qwen3-TTS."""
    return str(
        Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "qwen3_tts.yaml"
    )


@pytest.fixture(scope="module")
def omni_server():
    """Start vLLM-Omni server with CustomVoice model."""
    stage_config_path = get_stage_config()

    print(f"Starting OmniServer with model: {MODEL}")

    with OmniServer(
        MODEL,
        [
            "--stage-configs-path",
            stage_config_path,
            "--stage-init-timeout",
            "120",
            "--trust-remote-code",
            "--enforce-eager",
        ],
    ) as server:
        print("OmniServer started successfully")
        yield server
        print("OmniServer stopping...")

    print("OmniServer stopped")


def make_speech_request(
    host: str,
    port: int,
    text: str,
    voice: str = "vivian",
    language: str = "English",
    task_type: str | None = None,
    instructions: str | None = None,
    timeout: float = 120.0,
) -> httpx.Response:
    """Make a request to the /v1/audio/speech endpoint."""
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "input": text,
        "voice": voice,
        "language": language,
    }
    if task_type:
        payload["task_type"] = task_type
    if instructions:
        payload["instructions"] = instructions

    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


def verify_wav_audio(content: bytes) -> bool:
    """Verify that content is valid WAV audio data."""
    # WAV files start with "RIFF" header
    if len(content) < 44:  # Minimum WAV header size
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


# Minimum expected audio size for a short sentence (~1 second of 24kHz 16-bit mono WAV)
MIN_AUDIO_BYTES = 10000


class TestQwen3TTSCustomVoice:
    """E2E tests for Qwen3-TTS CustomVoice model."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_speech_english_basic(self, omni_server) -> None:
        """Test basic English TTS generation."""
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="Hello, how are you?",
            voice="vivian",
            language="English",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio content too small ({len(response.content)} bytes), expected at least {MIN_AUDIO_BYTES} bytes"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_speech_chinese_basic(self, omni_server) -> None:
        """Test basic Chinese TTS generation."""
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="你好，我是通义千问",
            voice="vivian",
            language="Chinese",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio content too small ({len(response.content)} bytes), expected at least {MIN_AUDIO_BYTES} bytes"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_speech_different_voices(self, omni_server) -> None:
        """Test TTS with different voice options."""
        voices = ["vivian", "ryan"]
        for voice in voices:
            response = make_speech_request(
                host=omni_server.host,
                port=omni_server.port,
                text="Testing voice selection.",
                voice=voice,
                language="English",
            )

            assert response.status_code == 200, f"Request failed for voice {voice}: {response.text}"
            assert verify_wav_audio(response.content), f"Invalid WAV for voice {voice}"

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_speech_binary_response_not_utf8_error(self, omni_server) -> None:
        """
        Regression test: Verify binary audio is returned, not UTF-8 error.

        This test ensures the multimodal_output property correctly retrieves
        audio from completion outputs, preventing the "TTS model did not
        produce audio output" error.
        """
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="This should return binary audio, not a JSON error.",
            voice="vivian",
            language="English",
        )

        # Should NOT be a JSON error response
        assert response.status_code == 200, f"Request failed: {response.text}"

        # Verify it's binary audio, not JSON
        try:
            # If this succeeds and starts with {"error", it's a bug
            text = response.content.decode("utf-8")
            assert not text.startswith('{"error"'), f"Got error response instead of audio: {text}"
        except UnicodeDecodeError:
            # This is expected - binary audio can't be decoded as UTF-8
            pass

        assert verify_wav_audio(response.content), "Response is not valid WAV audio"


class TestQwen3TTSAPIEndpoints:
    """Test API endpoint functionality."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_list_voices_endpoint(self, omni_server) -> None:
        """Test the /v1/audio/voices endpoint returns available voices."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/voices"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)

        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert isinstance(data["voices"], list)
        assert len(data["voices"]) > 0
        # Check some expected voices are present
        voices_lower = [v.lower() for v in data["voices"]]
        assert "vivian" in voices_lower or "ryan" in voices_lower

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_models_endpoint(self, omni_server) -> None:
        """Test the /v1/models endpoint returns loaded model."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/models"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
