"""
End-to-end diffusion coverage for FLUX.2-dev in online serving mode.

Coverage:
- CPU offload

This test verifies that FLUX.2-dev can be launched with CPU offload enabled,
accepts text-to-image requests through the OpenAI-compatible API, and returns
valid generated images with the requested resolution.

assert_diffusion_response validates successful generation and the expected
image resolution.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

MODEL = "black-forest-labs/FLUX.2-dev"
PROMPT = "A cinematic mountain landscape at sunrise, dramatic clouds, ultra-detailed, realistic photography."
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_flux_2_dev_feature_cases(model: str):
    """Return FLUX.2-dev diffusion feature cases for CPU offload."""

    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-cpu-offload",
                ],
            ),
            id="cpu_offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-cpu-offload",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_cfg_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_flux_2_dev_feature_cases(MODEL),
    indirect=True,
)
def test_flux_2_dev(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Validate FLUX.2-dev online serving with CPU offload."""

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
