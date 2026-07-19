# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online serving smoke test for DeepSeek Janus text-to-image generation."""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "deepseek-ai/Janus-1.3B"
PROMPT = "A scenic mountain lake at sunset"
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_janus_cases():
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=get_deploy_config_path("deepseek_janus_single_stage.yaml"),
            ),
            id="single_stage_default",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.diffusion
@pytest.mark.full_model
@pytest.mark.parametrize("omni_server", _get_janus_cases(), indirect=True)
def test_janus_single_stage(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 384,
            "width": 384,
            "guidance_scale": 5.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
