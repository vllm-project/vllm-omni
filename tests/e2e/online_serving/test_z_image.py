"""
Tests of common diffusion feature combinations in online serving mode
for Z-Image.

Coverage is intentionally limited to the minimal 4xL4 cases that
exercise Z-Image's supported parallel feature combinations:
- CacheDiT + FP8 + Ring=2 + TP=2
- TeaCache + FP8 + Ulysses=2 + Ring=2
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

MODEL = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A high-detail studio photo of an orange tabby cat sitting on a laptop keyboard."

FOUR_CARD_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=4)


def _get_diffusion_feature_cases():
    """Return the common Z-Image feature combinations that fit L4 CI."""
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
            ),
            id="default",
            marks=FOUR_CARD_MARKS,
        )
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(),
    indirect=True,
)
def test_basic_001(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Exercise supported Z-Image diffusion features in minimal CI cases."""
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(),
    indirect=True,
)
def test_different_sizes_001(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Exercise supported Z-Image diffusion features in minimal CI cases."""
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": [512, 512],
            "width": [512, 768],
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
