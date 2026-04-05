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

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

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
