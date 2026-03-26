"""
Tests of common diffusion feature combinations in online serving mode
for Z-Image.

Coverage is intentionally limited to the minimal 4xL4 cases that
exercise Z-Image's supported parallel feature combinations:
- CacheDiT + FP8 + Ring=2 + TP=2
- TeaCache + FP8 + Ulysses=2 + Ring=2
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
)
from tests.utils import hardware_marks

MODEL = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A high-detail studio photo of an orange tabby cat sitting on a laptop keyboard."

FOUR_CARD_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=4)


def _get_diffusion_feature_cases():
    """Return the common Z-Image feature combinations that fit L4 CI."""
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--quantization",
                    "fp8",
                    "--ring",
                    "2",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            id="parallel_cachedit_fp8_ring2_tp2",
            marks=FOUR_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--quantization",
                    "fp8",
                    "--ulysses-degree",
                    "2",
                    "--ring",
                    "2",
                ],
            ),
            id="parallel_teacache_fp8_ulysses2_ring2",
            marks=FOUR_CARD_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(),
    indirect=True,
)
def test_zimage(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Exercise supported Z-Image diffusion features in minimal CI cases."""
    request_config = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
