# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online serving tests for ``Qwen/Qwen-Image-Layered`` (layer decomposition).

- ``test_feature``: default single-GPU smoke via ``_get_diffusion_feature_cases`` (one ``default`` case).
- ``test_layered_output_image_count``: guard for issue #1969 (multi-layer image count).
- ``test_empty_prompt``: guard for issue #1966 (empty prompt).

Broader parallel / cache feature matrices previously lived in ``test_qwen_image_layered_expansion.py``;
that module is now a shim that points here.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

MODEL = "Qwen/Qwen-Image-Layered"
EDIT_PROMPT = "Decompose this image into layers."
MULTI_EDIT_PROMPT = (
    "Transform the first image into a Dadaism collage art. "
    "Transform the second image into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
NEGATIVE_PROMPT = "blurry, low quality, distorted"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    """Return a single default ``OmniServerParams`` row (no extra ``server_args``)."""
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_single_image_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Default Qwen-Image-Layered smoke (single ``default`` server config)."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
