# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox speech serving through the shared TTS test client."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = "ResembleAI/chatterbox-turbo"
TEXT = "The weather is nice today, perfect for a walk in the park."
DEPLOY = get_deploy_config_path("chatterbox_turbo.yaml")
SERVERS = [
    pytest.param(OmniServerParams(model=MODEL, stage_config_path=DEPLOY), id="v2"),
    pytest.param(
        OmniServerParams(model=MODEL, stage_config_path=modify_stage_config(DEPLOY, {"model_runner": "v1"})),
        id="v1",
    ),
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
def test_text_to_audio_001(omni_server, online_client):
    """The built-in voice works without reference encoders or a transcript."""
    online_client.send_audio_speech_request(
        {"model": MODEL, "input": TEXT, "voice": "default", "response_format": "wav", "timeout": 300.0}
    )


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
@pytest.mark.parametrize("response_format", ["wav", "pcm"])
def test_streaming_concurrent_audio(omni_server, online_client, response_format):
    """Four overlapping streams finish independently through the public API."""
    online_client.send_audio_speech_request(
        {
            "model": MODEL,
            "input": TEXT,
            "stream": True,
            "stream_format": "audio",
            "response_format": response_format,
            "timeout": 300.0,
        },
        request_num=4,
    )


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
def test_reference_voice_without_transcript(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": MODEL,
            "input": TEXT,
            "ref_audio": get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True),
            "response_format": "wav",
            "timeout": 300.0,
        }
    )
