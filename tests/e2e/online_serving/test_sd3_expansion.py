"""
E2E expansion tests for Stable Diffusion 3.5 medium model (nightly CI).
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

FOUR_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=4)
POSITIVE_PROMPT = "A serene mountain landscape at sunset"
NEGATIVE_PROMPT = "blurry, low quality, distorted"

MODEL = "stabilityai/stable-diffusion-3.5-medium"


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--cfg-parallel-size",
                    "2",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            id="cache_dit_cfg_tp",
            marks=[
                *FOUR_CARD_FEATURE_MARKS,
                pytest.mark.skip(reason="#3432"),
            ],
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_sd3_medium(omni_server: OmniServer, openai_client: OpenAIClientHandler):
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
