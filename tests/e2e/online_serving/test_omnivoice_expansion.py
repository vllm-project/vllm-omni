# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for OmniVoice TTS model via /v1/audio/speech endpoint.

Tests verify that the OmniVoice model generates valid audio when
accessed through the standard OpenAI-compatible speech API.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

try:
    from transformers import HiggsAudioV2TokenizerModel  # noqa: F401

    _HAS_VOICE_CLONE = True
except ImportError:
    _HAS_VOICE_CLONE = False

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

MODEL = "k2-fsa/OmniVoice"

STAGE_CONFIG = get_deploy_config_path("omnivoice.yaml")
EXTRA_ARGS = [
    "--trust-remote-code",
    "--disable-log-stats",
]
TEST_PARAMS = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG,
        server_args=EXTRA_ARGS,
    )
]

# Lower this in ``request_config`` via ``min_audio_bytes`` if a run produces legitimately short WAVs.
_DEFAULT_MIN_AUDIO_BYTES = 5000


def _get_ref_audio_b64() -> str:
    """Generate synthetic speech for reference audio.

    Returns:
        Base64 data URL string (data:audio/wav;base64,...)
    """
    audio_data = generate_synthetic_audio(duration=2, num_channels=1, sample_rate=24000)
    return f"data:audio/wav;base64,{audio_data['base64']}"


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceTTS:
    """E2E tests for OmniVoice TTS model."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice(self, omni_server, openai_client) -> None:
        """Test auto voice TTS generation (text only, no reference audio)."""
        request_config = {
            "model": omni_server.model,
            "input": "Hello, this is a test of the OmniVoice text to speech system.",
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)


@pytest.mark.skipif(not _HAS_VOICE_CLONE, reason="Voice cloning requires transformers>=5.3.0")
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceVoiceCloning:
    """E2E tests for OmniVoice voice cloning functionality."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_only(self, omni_server, openai_client) -> None:
        """Test voice cloning with ref_audio only (x_vector mode)."""
        request_config = {
            "model": omni_server.model,
            "input": "Hello, this is a voice cloning test.",
            "ref_audio": _get_ref_audio_b64(),
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_and_text(self, omni_server, openai_client) -> None:
        """Test voice cloning with ref_audio and ref_text (in-context mode)."""
        request_config = {
            "model": omni_server.model,
            "input": "Hello, this is a voice cloning test with in-context learning.",
            "ref_audio": _get_ref_audio_b64(),
            "ref_text": "This is the reference transcript.",
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_invalid_ref_audio_format(self, omni_server, openai_client) -> None:
        """Test that invalid ref_audio format returns a clear error."""
        request_config = {
            "model": omni_server.model,
            "input": "This should fail with invalid ref_audio.",
            "ref_audio": "not_a_valid_uri",
            "response_format": "wav",
            "timeout": 180.0,
            "status_code": (400, 422),
        }
        openai_client.send_audio_speech_request(request_config)
