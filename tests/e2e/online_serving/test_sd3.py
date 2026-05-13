"""
Tests for Stable Diffusion 3.5 medium model.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

FOUR_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=4)
POSITIVE_PROMPT = "A serene mountain landscape at sunset"
NEGATIVE_PROMPT = "blurry, low quality, distorted"


# For now, we only test one or two good configurations for this model to keep
# the CI light. The test cases focus on the most common feature combinations
# that provide good performance improvements.
def _get_diffusion_feature_cases(model: str):
    return [
        # Cache-DiT + CFG Parallel + Tensor Parallel
        pytest.param(
            OmniServerParams(
                model=model,
            ),
            id="default",
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(
        model="stabilityai/stable-diffusion-3.5-medium",
    ),
    indirect=True,
)
def test_text_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    messages = dummy_messages_from_mix_data(content_text=POSITIVE_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 28,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.5,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
