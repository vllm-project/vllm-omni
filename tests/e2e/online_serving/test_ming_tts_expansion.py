# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E expansion tests for Ming-omni-tts online serving (nightly CI).

Tests text-to-audio via /v1/audio/speech endpoint.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4704"),
]

MODEL = "inclusionAI/Ming-omni-tts-0.5B"
DEPLOY_CONFIG = get_deploy_config_path("ming_tts.yaml")

no_async_chunk_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            server_args=["--enforce-eager", "--no-async-chunk"],
        ),
        id="no_async_chunk",
    )
]

async_chunk_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            server_args=["--enforce-eager"],
        ),
        id="async_chunk",
    )
]


def get_prompt(prompt_type="zh"):
    prompts = {
        "zh": "今天天气真不错，适合出去散散步。",
        "zh_short": "这款产品的名字，叫变态坑爹牛肉丸。",
    }
    return prompts.get(prompt_type, prompts["zh"])


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", no_async_chunk_params, indirect=True)
def test_text_to_audio_non_streaming_001(omni_server, openai_client) -> None:
    """
    Deploy Setting: ming_tts.yaml with --no-async-chunk
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: two concurrent requests
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh"),
        "stream": False,
        "response_format": "wav",
        "timeout": 300.0,
    }
    openai_client.send_audio_speech_request(request_config, request_num=2)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", async_chunk_params, indirect=True)
def test_text_to_audio_streaming_001(omni_server, openai_client) -> None:
    """
    Deploy Setting: ming_tts.yaml (async_chunk=true)
    Input Modal: text + voice
    Output Modal: audio (streamed)
    Input Setting: stream=True
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh_short"),
        "voice": "灵小甄",
        "stream": True,
        "stream_format": "audio",
        "response_format": "wav",
        "timeout": 300.0,
    }
    openai_client.send_audio_speech_request(request_config)
