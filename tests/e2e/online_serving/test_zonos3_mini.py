# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Zonos3-Mini Speech API smoke with the default two-stage graph deployment."""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("ZONOS3_MODEL_PATH", "Zyphra/ZONOS3-Mini")


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("zonos3_mini.yaml"),
            env_dict={"FLA_TILELANG": "0"},
        )
    ],
    indirect=True,
)
def test_text_to_audio_001(omni_server, online_client) -> None:
    """Exercise prompt IPC, sampling, transfer and Codec PCM with real weights.

    Zonos3's strict companion loaders do not support the core-model harness's
    dummy weights. Advanced mode also validates the synthesized spoken content.
    """
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The weather is nice today, perfect for a walk in the park.",
            "seed": 42,
            "response_format": "wav",
            "min_audio_bytes": 44100,
            "max_new_tokens": 128,
            "timeout": 120.0,
            "transcript_language": "en",
        }
    )
