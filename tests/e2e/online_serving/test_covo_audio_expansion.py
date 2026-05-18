# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Covo-Audio-Chat model.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
    COVO_AUDIO_SYSTEM_PROMPT,
)

model = "tencent/Covo-Audio-Chat"
stage_config_path = get_deploy_config_path("covo_audio.yaml")

test_params = [
    pytest.param(
        OmniServerParams(
            model=model,
            stage_config_path=stage_config_path,
            server_args=["--trust-remote-code"],
        ),
        id="default",
    ),
]


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text + audio
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(2, 1, sample_rate=16000)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt={"role": "system", "content": COVO_AUDIO_SYSTEM_PROMPT},
        audio_data_url=audio_data_url,
        content_text="请回答这段音频里的问题。",
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "stream": False,
    }

    openai_client.send_omni_request(request_config)
