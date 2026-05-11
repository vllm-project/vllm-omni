"""
E2E online serving tests for Raon model with text input and audio/text output.

Raon is a 2-stage streaming pipeline:
  Stage 0: AR text + codec generation (thinker-talker fused)
  Stage 1: Audio codec-to-waveform decode
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import httpx
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, dummy_messages_from_mix_data

# Model identifier – placeholder for local checkpoint path.
MODEL = "/path/to/Raon"


def get_stage_config():
    """Get the CI stage config path for Raon."""
    return str(Path(__file__).parent.parent / "stage_configs" / "raon_ci.yaml")


stage_configs = [get_stage_config()]

# Create parameter combinations for model and stage config
test_params = [(MODEL, stage_config) for stage_config in stage_configs]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": ("You are a helpful assistant capable of generating speech."),
            }
        ],
    }


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of France? Answer in 20 words.",
        "tts": "Hello, this is a test of the Raon text-to-speech pipeline.",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def verify_wav_audio(content: bytes) -> bool:
    """Verify that content is valid WAV audio data."""
    if len(content) < 44:  # Minimum WAV header size
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


# Minimum expected audio size for a short sentence (~1 second of audio)
MIN_AUDIO_BYTES = 10000


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server with Raon model."""
    model, stage_config_path = request.param

    with OmniServer(
        model,
        [
            "--stage-configs-path",
            stage_config_path,
            "--stage-init-timeout",
            "120",
            "--disable-log-stats",
        ],
    ) as server:
        print("OmniServer started successfully")
        yield server
        print("OmniServer stopping...")

    print("OmniServer stopped")


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_stream(omni_server) -> None:
    """
    Test text-to-audio streaming via /v1/audio/speech endpoint.
    Deploy Setting: CI yaml (single GPU)
    Input Modal: text
    Output Modal: audio (streaming PCM)
    """
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": get_prompt("tts"),
        "voice": "default",
        "stream": True,
        "response_format": "pcm",
    }

    total_bytes = 0
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", url, json=payload) as response:
            assert response.status_code == 200, f"Request failed: {response.text}"
            for chunk in response.iter_bytes():
                total_bytes += len(chunk)

    assert total_bytes > MIN_AUDIO_BYTES, f"Streamed audio too small ({total_bytes} bytes)"


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text input processing and text output generation via OpenAI API.
    Deploy Setting: CI yaml (single GPU)
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text_only"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["paris"]},
    }

    openai_client.send_request(request_config)


class TestRaonSpeechEndpoint:
    """Tests for the /v1/audio/speech TTS endpoint."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    @pytest.mark.parametrize("omni_server", test_params, indirect=True)
    def test_speech_basic(self, omni_server) -> None:
        """Test basic TTS generation via /v1/audio/speech."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": "Hello, this is a Raon speech test.",
            "voice": "default",
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio content too small ({len(response.content)} bytes), expected at least {MIN_AUDIO_BYTES} bytes"
        )
