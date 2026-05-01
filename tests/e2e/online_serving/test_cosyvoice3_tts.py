# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for CosyVoice3 TTS model with voice cloning.

These tests verify the /v1/audio/speech endpoint works correctly with
the CosyVoice3 model, which requires reference audio for voice cloning.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# Official CosyVoice zero-shot prompt audio and its transcript. Vendored under
# tests/assets/ so the server does not depend on raw.githubusercontent.com being
# reachable at request time (same rationale as issue #3263 for Qwen3-TTS).
REF_AUDIO_URL = load_test_audio_data_url("cosyvoice3/zero_shot_prompt.wav")
REF_TEXT = "希望你以后能够做的比我还好呦。"


def get_stage_config(name: str = "cosyvoice3.yaml"):
    """Get the deploy config path for CosyVoice3."""
    return get_deploy_config_path(name)


def get_prompt(prompt_type="zh"):
    prompts = {
        "zh": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的感动让我热泪盈眶。",
        "en": "Hello, this is a voice cloning test with English text.",
    }
    return prompts.get(prompt_type, prompts["zh"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config(),
            server_args=["--trust-remote-code", "--disable-log-stats", "--no-async-chunk"],
        ),
        id="cosyvoice3",
    )
]

tts_async_chunk_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config(),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="cosyvoice3_async_chunk",
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_zh_001(omni_server, openai_client) -> None:
    """
    Test voice cloning TTS with Chinese text via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_async_chunk_server_params, indirect=True)
def test_voice_clone_zh_002(omni_server, openai_client) -> None:
    """
    Test voice cloning TTS with Chinese text via async_chunk streaming.
    Deploy Setting: cosyvoice3.yaml with default ``async_chunk: true``
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio (streamed)
    Input Setting: stream=True
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh"),
        "stream": True,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_en_001(omni_server, openai_client) -> None:
    """
    Test voice cloning TTS with English text via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)
