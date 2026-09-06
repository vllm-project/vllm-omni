"""
Online serving diffusion feature coverage for Ovis-Image.

Coverage:
- Ulysses-SP

assert_diffusion_response validates successful generation and the expected
image resolution.
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.utils import hardware_marks

MODEL = "AIDC-AI/Ovis-Image-7B"
PROMPT = "A cozy reading nook by a window, warm afternoon sunlight, realistic interior photography."
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_ovis_image_feature_cases(model: str):
    """Return Ovis-Image online serving cases for SP."""

    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--usp",
                    "2",
                ],
            ),
            id="ulysses_sp_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_ovis_image_feature_cases(MODEL),
    indirect=True,
)
def test_ovis_image(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Validate Ovis-Image online serving for Ulysses-SP."""

    messages = dummy_messages_from_mix_data(content_text=PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
