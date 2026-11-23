# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online Phonon speech API smoke test."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "gradium/phonon-31m-en"

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_server",
        [OmniServerParams(model=MODEL, stage_config_path=get_deploy_config_path("phonon.yaml"))],
        indirect=True,
    ),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_default_voice_speech(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "Hello, this is a short Phonon speech synthesis test.",
            "voice": "default",
            "response_format": "wav",
            "min_audio_bytes": 24000,
            "transcript_language": "en",
            "seed": 42,
            "max_new_tokens": 250,
        }
    )
