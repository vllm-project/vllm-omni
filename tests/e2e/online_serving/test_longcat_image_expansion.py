"""
Recommended tests of diffusion features that are available in online serving mode
and are supported by the following model:
- LongCat-Image: text-to-image with single prompt input
Coverage:
- CPU offloading (model-level sequential offload via --enable-cpu-offload)
- Cache-DiT
- SP (Ulysses)

This validates:
 - Successful image generation at the expected 768x1344 resolution with recommended feature combinations
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

TEXT_TO_IMAGE_PROMPT = (
    "A cinematic illustration of a cat typing on a silver laptop, soft window light, highly detailed."
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, oversaturated"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    """Return diffusion feature cases for LongCat-Image."""
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--enable-cpu-offload"],
            ),
            id="single_card_001",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ulysses-degree",
                    "2",
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("meituan-longcat/LongCat-Image"),
    indirect=True,
)
def test_longcat_image(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test the recommended feature combinations for LongCat-Image."""
    messages = dummy_messages_from_mix_data(content_text=TEXT_TO_IMAGE_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 768,
            "width": 1344,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
