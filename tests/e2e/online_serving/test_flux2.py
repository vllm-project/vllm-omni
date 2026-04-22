"""
Tests for Flux2 Klein; currently Dev is implemented separately,
but ideally these models will fold together in the future.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

FOUR_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=4)
POSITIVE_PROMPT = "A cat sitting on a windowsill"
NEGATIVE_PROMPT = "blurry, low quality"


# For now, we only test a good configuration for this model to keep the CI
# light, but in the near future may explore using random models to check
# cross-feature compatibility more generally.
def _get_diffusion_feature_cases(model: str):
    return [
        # FP8 / Hybrid sequence parallelism
        pytest.param(
            OmniServerParams(
                model=model,
            ),
            id="default",
            marks=FOUR_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(
        model="black-forest-labs/FLUX.2-klein-4B",
    ),
    indirect=True,
)
def test_text_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    messages = dummy_messages_from_mix_data(content_text=POSITIVE_PROMPT)
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
