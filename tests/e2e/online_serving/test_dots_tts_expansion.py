# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online expansion tests for dots.tts(text-only).
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "dots-studio/dots.tts-soar"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
_MIN_AUDIO_BYTES = 40_000
MAX_CONCURRENT = 4


def get_prompt(prompt_type="text"):
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("dots_tts.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="dots_tts",
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_only_streaming_001(omni_server, openai_client) -> None:
    """
    Test text-only text-to-audio via OpenAI API.
    Deploy Setting: dots_tts.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: few requests (max_num_seqs=4)
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "default",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_only_nonstreaming_pcm_001(omni_server, openai_client) -> None:
    """
    Test text-only non-stream PCM output via OpenAI API.
    Deploy Setting: dots_tts.yaml
    Input Modal: text
    Output Modal: audio (pcm)
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "response_format": "pcm",
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "voice": "default",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        "min_hnr_db": -2.0,
    }
    openai_client.send_audio_speech_request(request_config)
