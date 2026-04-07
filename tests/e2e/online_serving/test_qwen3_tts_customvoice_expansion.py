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

import pytest

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def get_stage_config(name: str = "qwen3_tts.yaml"):
    """Get the stage config path from vllm_omni model_executor stage_configs."""
    return str(Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / name)


def get_prompt(prompt_type="english"):
    """Text prompt for text-to-audio tests (same as test_qwen3_omni - beijing test case)."""
    prompts = {
        "english": "Paris is a city in France. It has a tall tower by the river. The streets are old and small. You can walk and see art and food. The sky is light and soft. It is a calm place to rest.",
        "chinese": "北京，中国的首都，是一座融合了长城等历史地点与现代建筑的国际化大都市，充满了独特的文化与活力",
    }
    return prompts.get(prompt_type, prompts["english"])


def get_max_batch_size(size_type="few"):
    """Batch size for concurrent requests (same as test_qwen3_omni)."""
    batch_sizes = {"few": 3, "medium": 100, "large": 256}
    return batch_sizes.get(size_type)


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="async_chunk",
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config("qwen3_tts_no_async_chunk.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="no_async_chunk",
    ),
]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False, voice=eric
    Datasets: few requests
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "uncle_fu",
    }

    # Retry once on assertion failures from transcript similarity / gender checks (flaky ASR or estimators).
    for attempt in range(2):
        try:
            openai_client.send_audio_speech_request(request_config, request_num=get_max_batch_size())
            break
        except AssertionError:
            if attempt == 0:
                continue
            raise


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_002(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False, language=chinese
    Datasets: few requests
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "Serena",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_003(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False, language=chinese
    Datasets: few requests
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "SERENA",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_language_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False, language=chinese
    Datasets: few requests
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("chinese"),
        "stream": False,
        "response_format": "wav",
        "task_type": "CustomVoice",
        "voice": "vivian",
    }

    openai_client.send_audio_speech_request(request_config)
