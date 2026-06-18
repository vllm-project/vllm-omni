# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online expansion tests for VoxCPM2 (voice clone, PCM, concurrency).

Voice clone uses ``ref_audio`` only (no ``ref_text``). Reference clip is
vendored under tests/assets/qwen3_tts/clone_2.wav — see test_qwen3_tts_base.py.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "openbmb/VoxCPM2"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
_MIN_AUDIO_BYTES = 40_000
MAX_CONCURRENT = 4

REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")


def get_prompt(prompt_type="text"):
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("voxcpm2.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="voxcpm2",
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_streaming_001(omni_server, openai_client) -> None:
    """
    Test voice-clone text-to-audio via OpenAI API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text + ref_audio
    Output Modal: audio
    Input Setting: stream=True
    Datasets: few requests (max_num_seqs=4)
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "default",
        "ref_audio": REF_AUDIO_URL,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
@pytest.mark.skip(reason="#4335")
def test_response_format_001(omni_server, openai_client) -> None:
    """
    Test voice-clone non-stream PCM output via OpenAI API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text + ref_audio
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
        "ref_audio": REF_AUDIO_URL,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config)
