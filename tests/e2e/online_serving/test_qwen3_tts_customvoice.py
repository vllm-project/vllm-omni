# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def get_prompt(prompt_type="text"):
    """Text prompt for text-to-audio tests (same as test_qwen3_omni - beijing test case)."""
    prompts = {
        "text": "Beijing, China's capital, blends ancient wonders like the Great Wall with modern marvels. This vibrant metropolis offers rich culture, delicious Peking duck, and endless exploration opportunities.",
    }
    return prompts.get(prompt_type, prompts["text"])


def get_max_batch_size(size_type="few"):
    """Batch size for concurrent requests (same as test_qwen3_omni)."""
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="async_chunk",
    )
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: few requests
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "vivian",
    }

    openai_client.send_audio_speech_request(request_config, request_num=get_max_batch_size())


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "vivian",
    }

    openai_client.send_audio_speech_request(request_config)
