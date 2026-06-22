"""
Tests for Stable Diffusion XL Base 1.0 model.

Validates single-GPU text-to-image generation with CFG (guidance_scale > 1).
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"})
POSITIVE_PROMPT = "A serene mountain landscape at sunset, photorealistic"
NEGATIVE_PROMPT = "blurry, low quality, distorted"


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
            ),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(
        model="stabilityai/stable-diffusion-xl-base-1.0",
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
            "num_inference_steps": 30,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 7.5,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
