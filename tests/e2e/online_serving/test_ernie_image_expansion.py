# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for ErnieImage text-to-image generation via online serving.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

T2I_PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"
MODEL = "baidu/ERNIE-Image"
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--tensor-parallel-size",
                    "2",
                    "--enable-cpu-offload",
                ],
            ),
            id="tp_2_cpu_offload",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_ernie_image(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test each diffusion feature with ERNIE-Image (text-to-image), one feature per case."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)
