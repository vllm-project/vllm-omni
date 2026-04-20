"""
Recommended tests of diffusion features that are available in online serving mode
and are supported by the following model:
- LongCat-Image-Edit: image-to-image edit with a single image + single edit prompt input
Coverage:
- CPU offloading (model-level sequential offload via --enable-cpu-offload)
- Cache-DiT
- SP (Ulysses)

This validates:
 - Successful image generation at the expected 1024x1024 resolution with recommended feature combinations
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

EDIT_PROMPT = "Transform this modern image into a cinematic animation style with vibrant colors and soft lighting."
NEGATIVE_PROMPT = "blurry, low quality, distorted, oversaturated"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    """Return diffusion feature cases for LongCat-Image-Edit."""
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
    _get_diffusion_feature_cases("meituan-longcat/LongCat-Image-Edit"),
    indirect=True,
)
def test_longcat_image_edit(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test the recommended feature combinations for LongCat-Image-Edit."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
