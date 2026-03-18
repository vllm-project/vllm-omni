"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following model:
- LongCat-Image: text-to-image with single prompt input
Coverage:
- CPU offloading
- Cache-DIT
- SP (Ulysses & Ring)
- CFG-Parallel
- Tensor-Parallel
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.utils import hardware_marks

TEXT_TO_IMAGE_PROMPT = (
    "A cinematic illustration of a cat typing on a silver laptop, soft window light, highly detailed."
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, oversaturated"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-cpu-offload",
                ],
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
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ring",
                    "2",
                ],
            ),
            id="parallel_002",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_003",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            id="parallel_004",
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
    """Test all supported diffusion features with LongCat-Image in regular text-to-image scenarios."""
    messages = dummy_messages_from_mix_data(content_text=TEXT_TO_IMAGE_PROMPT)

    # CFG parallel is only activated when a negative prompt and guidance_scale > 1.0 are both present
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
