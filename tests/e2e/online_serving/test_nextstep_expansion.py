"""
Online serving E2E for NextStep-1.1 text-to-image (tensor parallel).
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

# L4: 4 GPUs + TP=4; XPU B60: 2 cards (use num_cards={"cuda": 4, "xpu": 4} if needed)
FOUR_CARD_MARKS = hardware_marks(
    res={"cuda": "L4", "xpu": "B60"},
    num_cards={"cuda": 2, "xpu": 2},
)

POSITIVE_PROMPT = "A small red barn in a snowy field, simple illustration."
NEGATIVE_PROMPT = "blurry, low quality"

_DEFAULT_MODEL = "stepfun-ai/NextStep-1.1"


def _get_diffusion_feature_cases(model: str):
    """Single online config: TP=4, explicit pipeline class."""
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--tensor-parallel-size",
                    "2",
                    "--model-class-name",
                    "NextStep11Pipeline",
                ],
            ),
            id="nextstep_tp4_pipeline",
            marks=FOUR_CARD_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(model=os.environ.get("VLLM_TEST_NEXTSTEP_MODEL", _DEFAULT_MODEL)),
    indirect=True,
)
def test_nextstep_11(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    messages = dummy_messages_from_mix_data(content_text=POSITIVE_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "guidance_scale_2": 1.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
