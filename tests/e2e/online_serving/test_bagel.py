"""
Online serving tests for ``ByteDance-Seed/BAGEL-7B-MoT`` (text-to-image via chat completions).

This module currently exposes a single parametrized case (``default``): default
``OmniServerParams`` (no extra ``server_args``). Multi-feature matrices (TeaCache,
Cache-DiT, parallel SP/TP, etc.) live in ``test_bagel_expansion.py``.

Responses are validated inside ``OpenAIClientHandler.send_diffusion_request`` via
``assert_diffusion_response`` (success path and image content); request uses
512×512, ``num_inference_steps=2``, and classifier-free guidance fields in
``extra_body``.
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
    generate_synthetic_image,
)
from tests.utils import hardware_marks

PROMPT = "A futuristic city skyline at twilight, cyberpunk style, ultra-detailed, high resolution."
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark"


def _get_diffusion_feature_cases(model: str):
    """Return parametrized ``OmniServerParams`` rows for this file.

    Only the default single-GPU case is registered here; extend the list to add
    more ``pytest.param(..., id=..., marks=...)`` rows when new server flags are
    needed.
    """
    return [
        pytest.param(
            OmniServerParams(
                model=model,
            ),
            id="default",
            marks=hardware_marks(res={"cuda": "H100"}),
        )
    ]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("ByteDance-Seed/BAGEL-7B-MoT"),
    indirect=True,
)
def test_text_to_image_001(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Default Bagel T2I smoke: chat completion with text-only ``messages`` and image ``extra_body``.

    Marked with both ``core_model`` and ``advanced_model`` so the same case can
    run under L2-style or heavier CI selections; feature-depth coverage remains
    in ``test_bagel_expansion.py``.
    """
    messages = dummy_messages_from_mix_data(content_text=PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            # Enable CFG for models that use classifier-free guidance
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("ByteDance-Seed/BAGEL-7B-MoT"),
    indirect=True,
)
def test_image_to_image_001(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Default Bagel T2I smoke: chat completion with text-only ``messages`` and image ``extra_body``.

    Marked with both ``core_model`` and ``advanced_model`` so the same case can
    run under L2-style or heavier CI selections; feature-depth coverage remains
    in ``test_bagel_expansion.py``.
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(content_text=PROMPT, image_data_url=image_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            # Enable CFG for models that use classifier-free guidance
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
