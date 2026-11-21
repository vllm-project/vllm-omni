# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online serving test for Confucius4-TTS via /v1/audio/speech."""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = os.environ.get(
    "CONFUCIUS4_TTS_MODEL_PATH",
    "netease-youdao/Confucius4-TTS",
)
REF_AUDIO_URL = get_asset_path("indextts2/ref_audio.wav", as_data_url=True)

tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("confucius4_tts.yaml"),
            server_args=["--disable-log-stats"],
            stage_init_timeout=600,
        ),
        id="confucius4_tts",
    )
]


@hardware_test(res={"cuda": "H200"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_english(omni_server, online_client) -> None:
    """Generate a non-streaming WAV from text and one reference clip."""
    request_config = {
        "model": omni_server.model,
        "input": "Hello, this is a Confucius4 text to speech test.",
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "extra_params": {"lang": "en"},
        "min_audio_bytes": 1024,
        "sample_rate": 22050,
        "timeout": 300.0,
    }
    online_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "H200"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_request_finished_on_prefill_returns_audio(omni_server, online_client) -> None:
    """max_new_tokens=1 ends Stage 0 on its prefill step; Stage 1 must still get input."""
    request_config = {
        "model": omni_server.model,
        "input": "Hello.",
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "extra_params": {"lang": "en"},
        "max_new_tokens": 1,
        "sample_rate": 22050,
        # Without the conditioning on the prefill step this waits for the
        # 600 s stage-input deadline instead of returning.
        "timeout": 60.0,
    }
    online_client.send_audio_speech_request(request_config)
