# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for MOSS-TTS-Nano model with text input and audio output.

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

MODEL = "OpenMOSS-Team/MOSS-TTS-Nano"


def get_prompt(prompt_type="text"):
    """Text prompt for text-to-audio tests."""
    prompts = {
        "text": "Hello, this is a test of MOSS-TTS-Nano text to speech synthesis.",
        "chinese": "你好，这是MOSS-TTS-Nano的语音合成测试。",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("moss_tts_nano.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="moss_tts_nano",
    )
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio (48 kHz)
    Input Setting: stream=False, voice=Junhao
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "voice": "Junhao",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, openai_client) -> None:
    """
    Test streaming text-to-audio via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio (48 kHz, PCM stream)
    Input Setting: stream=True, voice=Ava
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "response_format": "pcm",
        "voice": "Ava",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_003(omni_server, openai_client) -> None:
    """
    Test Chinese text-to-audio via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text (Chinese)
    Output Modal: audio (48 kHz)
    Input Setting: stream=False, voice=Junhao
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("chinese"),
        "stream": False,
        "response_format": "wav",
        "voice": "Junhao",
    }

    openai_client.send_audio_speech_request(request_config)
