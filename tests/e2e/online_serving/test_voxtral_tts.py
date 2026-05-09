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

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "mistralai/Voxtral-4B-TTS-2603"

STAGE_CONFIG = get_deploy_config_path("voxtral_tts.yaml")
EXTRA_ARGS = ["--trust-remote-code", "--enforce-eager", "--disable-log-stats"]
TEST_PARAMS = [OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG, server_args=EXTRA_ARGS)]


def verify_wav_audio(content: bytes) -> bool:
    """Verify that content is valid WAV audio data."""
    if len(content) < 44:
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


# Minimum expected audio size for a short sentence (~1 second of 24kHz 16-bit mono WAV)
_MIN_AUDIO_BYTES_BASIC = 10000


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestVoxtralTTSFixedVoice:
    """E2E tests for Voxtral TTS model."""

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_english_basic(self, omni_server, openai_client) -> None:
        """Test basic English TTS generation."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello, how are you?",
                "voice": "casual_female",
                "language": "English",
                "response_format": "wav",
                "timeout": 120.0,
                "min_audio_bytes": _MIN_AUDIO_BYTES_BASIC,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_english_streaming(self, omni_server, openai_client) -> None:
        """Test basic streaming English TTS generation (PCM via streaming API)."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello, how are you?",
                "voice": "casual_female",
                "language": "English",
                "stream": True,
                "response_format": "pcm",
                "timeout": 120.0,
                "min_audio_bytes": _MIN_AUDIO_BYTES_BASIC,
            }
        )

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_different_voices(self, omni_server, openai_client) -> None:
        """Test TTS with different voice presets."""
        voices = ["casual_female", "neutral_male"]
        for voice in voices:
            openai_client.send_audio_speech_request(
                {
                    "model": omni_server.model,
                    "input": "Testing voice selection.",
                    "voice": voice,
                    "language": "English",
                    "response_format": "wav",
                    "timeout": 120.0,
                }
            )

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_invalid_voice_rejected(self, omni_server, openai_client) -> None:
        """Request with a non-existent voice should be rejected (HTTP 200 with JSON or 4xx, depending on server)."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "This voice does not exist.",
                "voice": "nonexistent_voice_xyz",
                "language": "English",
                "response_format": "wav",
                "timeout": 120.0,
                "err_message": "Invalid speaker",
            }
        )
