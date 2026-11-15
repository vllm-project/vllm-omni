# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight LFM3 speech serving, including concurrent streaming requests."""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("LFM3_OMNI_MODEL_PATH", "LiquidAI/LFM3-Omni-2B")
SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("lfm3_omni.yaml"),
    server_args=[
        "--trust-remote-code",
        "--async-chunk",
        "--limit-mm-per-prompt",
        '{"image":0,"audio":0,"video":0}',
    ],
)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
@pytest.mark.parametrize("stream", [False, True])
def test_speech_response_and_concurrent_streaming(omni_server, openai_client, stream):
    request = {
        "model": omni_server.model,
        "input": "The quick brown fox jumps over the lazy dog.",
        "seed": 42,
        "extra_params": {"audio_temperature": 0.0},
        "max_new_tokens": 96,
        "response_format": "pcm" if stream else "wav",
        "stream": stream,
        "expected_sample_rate": 24000,
        "min_audio_bytes": 24000,
        "timeout": 240,
    }
    if stream:
        request["stream_format"] = "audio"
    openai_client.send_audio_speech_request(request, request_num=2 if stream else 1)
