# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online serving tests for IndexTTS2 via /v1/audio/speech endpoint.

Two-stage pipeline: GPT AR → S2Mel + BigVGAN. Output is 22050 Hz mono WAV.
Covers: basic TTS and the speech endpoint's stream=True compatibility path.
"""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "IndexTeam/IndexTTS-2"

REF_AUDIO_URL = load_test_audio_data_url("indextts2/ref_audio.wav")

tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("indextts2.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="indextts2",
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_basic_english(omni_server, openai_client) -> None:
    """
    Basic English TTS: text + ref_audio, no emotion.
    Deploy Setting: default yaml
    Input Modal: text + reference audio
    Output Modal: audio (22050 Hz, WAV)
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": "Hello, this is a voice cloning demo.",
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "min_audio_bytes": 1024,
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_streaming(omni_server, openai_client) -> None:
    """
    stream=True compatibility: IndexTTS2 itself is non-async-chunking, but the
    OpenAI speech endpoint should still accept the streaming request path and
    return a valid PCM response.

    Deploy Setting: default yaml
    Input Modal: text + reference audio
    Output Modal: audio (22050 Hz, PCM)
    Input Setting: stream=True
    """
    request_config = {
        "model": omni_server.model,
        "input": "Hello, this is a voice cloning demo.",
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
        "ref_audio": REF_AUDIO_URL,
        "min_audio_bytes": 1024,
    }
    openai_client.send_audio_speech_request(request_config)
