"""
Online serving diffusion feature coverage for OmniGen2.

Coverage:
- Ulysses-SP
- CFG-Parallel

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

MODEL = "OmniGen2/OmniGen2"
PROMPT = "A sleek concept car parked beside a glass building at sunrise, cinematic lighting, ultra detailed."
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"})
PARALLEL_FEATURE_MARKS = {
    "usp": hardware_marks(res={"cuda": "L4"}, num_cards=3),
    "cfg_parallel": hardware_marks(res={"cuda": "L4"}, num_cards=2),
}


def _get_omnigen2_feature_cases(model: str):
    """Return OmniGen2 online serving cases for SP, and CFG-Parallel."""

    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--usp",
                    "3",
                ],
            ),
            id="ulysses_sp_3",
            marks=PARALLEL_FEATURE_MARKS["usp"],
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="cfg_parallel_2",
            marks=PARALLEL_FEATURE_MARKS["cfg_parallel"],
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_omnigen2_feature_cases(MODEL),
    indirect=True,
)
def test_omnigen2(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Validate OmniGen2 online serving for SP, and CFG-Parallel."""

    messages = dummy_messages_from_mix_data(content_text=PROMPT)

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
