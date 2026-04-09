"""
End-to-end diffusion coverage for FLUX.2-dev in online serving mode.

Coverage:
- Cache-DiT cache acceleration backend
- CPU offload

This test verifies that FLUX.2-dev can be launched with the Cache-DiT backend
and CPU offload enabled, accepts text-to-image requests through the
OpenAI-compatible API, and returns valid generated images with the requested
resolution.

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

MODEL = "black-forest-labs/FLUX.2-dev"
PROMPT = "A cinematic mountain landscape at sunrise, dramatic clouds, ultra-detailed, realistic photography."
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_flux_2_dev_feature_cases(model: str):
    """Return FLUX.2-dev diffusion feature cases for Cache-DiT + CPU offload."""

    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-cpu-offload",
                ],
            ),
            id="cache_dit_cpu_offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
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
    """Validate FLUX.2-dev online serving with Cache-DiT and CPU offload."""

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
