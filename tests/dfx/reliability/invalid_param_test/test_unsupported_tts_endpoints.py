# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unsupported generation requests must not kill a speech model's engine."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model="fishaudio/drama-3",
                stage_config_path=get_deploy_config_path("fish_omni.yaml"),
                server_args=["--disable-log-stats"],
                stage_init_timeout=900,
            ),
            id="drama-3",
        ),
    ],
    indirect=True,
)
def test_chat_rejection_preserves_speech(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    speech_request = {
        "model": omni_server.model,
        "input": "The weather is nice today, perfect for a walk in the park.",
        "response_format": "wav",
        "timeout": 300,
        "min_audio_bytes": 40_000,
    }
    online_client.send_audio_speech_request(speech_request)

    for stream in (False, True):
        online_client.send_chat_completions_http_request(
            {
                "json": {
                    "model": omni_server.model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                    "stream": stream,
                },
            },
            err_code=400,
            err_message="Use /v1/audio/speech instead.",
        )
        online_client.send_health_http_request()
        online_client.send_audio_speech_request(speech_request)
