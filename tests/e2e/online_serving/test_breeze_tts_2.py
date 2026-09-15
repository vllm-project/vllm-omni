# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E Online tests for Breeze-TTS-2 with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks. Breeze-TTS-2 plain mode needs only
``input`` and a speaker tag (``voice`` defaults to ``S0``); reference
cloning requires ``ref_audio`` + ``ref_text`` and is not exercised here.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "BreezeBlue/Breeze-TTS-2"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
MAX_CONCURRENT = 4

# ~0.5 s of 24 kHz mono PCM_16 in WAV (~24k payload + header).
_MIN_AUDIO_BYTES = 40_000


def get_prompt(prompt_type="text"):
    """English prompt for zero-shot TTS."""
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("breeze_tts_2.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="breeze_tts_2",
    )
]


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, online_client) -> None:
    """
    Test plain text-to-audio via OpenAI API.
    Deploy Setting: breeze_tts_2.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: few requests (max_num_seqs=8)
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "S0",
        "sample_rate": 24000,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    online_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, online_client) -> None:
    """
    Test streaming text-to-audio via OpenAI API (async-chunk SSE deltas).
    Deploy Setting: breeze_tts_2.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "S0",
        "sample_rate": 24000,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    online_client.send_audio_speech_request(request_config)
