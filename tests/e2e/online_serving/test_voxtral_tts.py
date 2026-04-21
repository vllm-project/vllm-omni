# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Voxtral TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import httpx
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "mistralai/Voxtral-4B-TTS-2603"

STAGE_CONFIG = get_deploy_config_path("voxtral_tts.yaml")
EXTRA_ARGS = ["--trust-remote-code", "--enforce-eager", "--disable-log-stats"]
TEST_PARAMS = [OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG, server_args=EXTRA_ARGS)]


def make_speech_request(
    host: str,
    port: int,
    text: str,
    voice: str = "casual_female",
    language: str = "English",
    timeout: float = 120.0,
) -> httpx.Response:
    """Make a request to the /v1/audio/speech endpoint."""
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "input": text,
        "voice": voice,
        "language": language,
    }

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


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestVoxtralTTSFixedVoice:
    """E2E tests for Voxtral TTS model."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_english_basic(self, omni_server) -> None:
        """Test basic English TTS generation."""
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="Hello, how are you?",
            voice="casual_female",
            language="English",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio content too small ({len(response.content)} bytes), expected at least {MIN_AUDIO_BYTES} bytes"
        )

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_different_voices(self, omni_server) -> None:
        """Test TTS with different voice presets."""
        voices = ["casual_female", "neutral_male"]
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

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_invalid_voice_rejected(self, omni_server) -> None:
        """Request with a non-existent voice should return an error response."""
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="This voice does not exist.",
            voice="nonexistent_voice_xyz",
            language="English",
        )

        # vLLM returns ErrorResponse as JSON with HTTP 200;
        # verify the body is a JSON error, not valid WAV audio.
        data = response.json()
        assert "error" in data or "message" in data, f"Expected error response for invalid voice, got: {data}"
        assert not verify_wav_audio(response.content), "Should not return valid audio for invalid voice"

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
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
            voice="casual_female",
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


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestVoxtralTTSAPIEndpoints:
    """Test API endpoint functionality."""

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
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

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_models_endpoint(self, omni_server) -> None:
        """Test the /v1/models endpoint returns loaded model."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/models"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
