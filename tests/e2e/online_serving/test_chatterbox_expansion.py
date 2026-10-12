# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Nightly encoded-response coverage for Chatterbox's shared speech API."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

SERVERS = [
    pytest.param(
        OmniServerParams(model=model, stage_config_path=get_deploy_config_path(deploy)),
        id=variant,
    )
    for variant, model, deploy in (
        ("turbo", "ResembleAI/chatterbox-turbo", "chatterbox_turbo.yaml"),
        ("original", "ResembleAI/chatterbox", "chatterbox.yaml"),
    )
]


@pytest.mark.full_model
@pytest.mark.tts
@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
@pytest.mark.parametrize("response_format", ["mp3", "flac"])
def test_encoded_audio(omni_server, online_client, response_format):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The weather is nice today, perfect for a walk in the park.",
            "voice": "default",
            "response_format": response_format,
            "timeout": 300.0,
        }
    )
