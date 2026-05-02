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

from pathlib import Path

import httpx
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams

try:
    from transformers import HiggsAudioV2TokenizerModel  # noqa: F401

    _HAS_VOICE_CLONE = True
except ImportError:
    _HAS_VOICE_CLONE = False

MODEL = "k2-fsa/OmniVoice"

STAGE_CONFIG = str(
    Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "omnivoice.yaml"
)
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

MIN_AUDIO_BYTES = 5000


def _get_ref_audio_b64() -> str:
    """Generate synthetic speech for reference audio.

    Returns:
        Base64 data URL string (data:audio/wav;base64,...)
    """
    audio_data = generate_synthetic_audio(duration=2, num_channels=1, sample_rate=24000)
    return f"data:audio/wav;base64,{audio_data['base64']}"


def make_speech_request(
    host: str,
    port: int,
    text: str,
    timeout: float = 180.0,
) -> httpx.Response:
    """Make a request to the /v1/audio/speech endpoint for OmniVoice."""
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {"input": text}

    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


def verify_wav_audio(content: bytes) -> bool:
    """Verify that content is valid WAV audio data."""
    if len(content) < 44:
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceTTS:
    """E2E tests for OmniVoice TTS model."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice(self, omni_server) -> None:
        """Test auto voice TTS generation (text only, no reference audio)."""
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="Hello, this is a test of the OmniVoice text to speech system.",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio too small ({len(response.content)} bytes), expected > {MIN_AUDIO_BYTES}"
        )


def make_voice_clone_request(
    host: str,
    port: int,
    text: str,
    ref_audio_b64: str,
    ref_text: str | None = None,
    timeout: float = 180.0,
) -> httpx.Response:
    """Make a voice cloning request to the /v1/audio/speech endpoint.

    Args:
        host: Server host
        port: Server port
        text: Text to synthesize
        ref_audio_b64: Base64-encoded reference audio data URL
        ref_text: Optional transcript of reference audio
        timeout: Request timeout in seconds

    Returns:
        httpx.Response object
    """
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "input": text,
        "ref_audio": ref_audio_b64,
    }
    if ref_text:
        payload["ref_text"] = ref_text

    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


@pytest.mark.skipif(not _HAS_VOICE_CLONE, reason="Voice cloning requires transformers>=5.3.0")
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceVoiceCloning:
    """E2E tests for OmniVoice voice cloning functionality."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_only(self, omni_server) -> None:
        """Test voice cloning with ref_audio only (x_vector mode)."""
        ref_audio_b64 = _get_ref_audio_b64()

        response = make_voice_clone_request(
            host=omni_server.host,
            port=omni_server.port,
            text="Hello, this is a voice cloning test.",
            ref_audio_b64=ref_audio_b64,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio too small ({len(response.content)} bytes), expected > {MIN_AUDIO_BYTES}"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_and_text(self, omni_server) -> None:
        """Test voice cloning with ref_audio and ref_text (in-context mode)."""
        ref_audio_b64 = _get_ref_audio_b64()
        ref_text = "This is the reference transcript."

        response = make_voice_clone_request(
            host=omni_server.host,
            port=omni_server.port,
            text="Hello, this is a voice cloning test with in-context learning.",
            ref_audio_b64=ref_audio_b64,
            ref_text=ref_text,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio too small ({len(response.content)} bytes), expected > {MIN_AUDIO_BYTES}"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_invalid_ref_audio_format(self, omni_server) -> None:
        """Test that invalid ref_audio format returns a clear error."""
        response = make_voice_clone_request(
            host=omni_server.host,
            port=omni_server.port,
            text="This should fail with invalid ref_audio.",
            ref_audio_b64="not_a_valid_uri",
        )

        assert response.status_code in (400, 422), (
            f"Expected 400/422 for invalid ref_audio format, got {response.status_code}"
        )
