# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online serving tests for the Qwen-Image-Edit family (image-to-image via chat completions).

- ``test_single_image_to_image_001``: ``Qwen/Qwen-Image-Edit`` — one reference image, fixed 512×512.
- ``test_multi_images_to_image_001``: ``Qwen/Qwen-Image-Edit-2509`` — two reference images, fixed 512×512.
- ``test_different_sizes_001``: ``Qwen/Qwen-Image-Edit-2509`` only, ``advanced_model`` — mixed input
  resolutions; ``extra_body`` uses per-output ``width``/``height`` lists; the test client sends one
  scalar-size request per list index in parallel and merges images (see ``OpenAIClientHandler.send_diffusion_request``).

``_get_diffusion_feature_cases`` registers a single ``default`` ``OmniServerParams`` row per model.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_qwen_image_edit.py -m "core_model and diffusion" --run-level=core_model
    pytest -s -v e2e/online_serving/test_qwen_image_edit.py -m "advanced_model and diffusion" --run-level=advanced_model
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

EDIT_PROMPT = "Transform this modern, geometrist image into a Vincent van Gogh style impressionist painting."
MULTI_EDIT_PROMPT = (
    "Transform the first image into a Dadaism collage art. "
    "Transform the second image into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
NEGATIVE_PROMPT = "blurry, low quality, modern, geometrist"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    """Return one ``default`` ``OmniServerParams`` row for ``model`` (no extra ``server_args``)."""
    return [
        pytest.param(
            OmniServerParams(
                model=model,
            ),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit"),
    indirect=True,
)
def test_single_image_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Single-reference edit smoke for ``Qwen/Qwen-Image-Edit`` (CFG when negative prompt + ``true_cfg_scale`` > 1)."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    # CFG parallel is only activated when a negative prompt and true_cfg_scale > 1.0 are both present
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


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit-2509"),
    indirect=True,
)
def test_multi_images_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Two-reference edit smoke for ``Qwen/Qwen-Image-Edit-2509``."""

    image_data_url_list = [f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}" for _ in range(2)]

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url_list, content_text=MULTI_EDIT_PROMPT)

    # CFG parallel is only activated when a negative prompt and true_cfg_scale > 1.0 are both present
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


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit-2509"),
    indirect=True,
)
def test_different_sizes_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Multi-reference edit with distinct input resolutions; per-output ``width``/``height`` as lists (client splits into concurrent scalar-size calls)."""
    image_data_url_list = [
        f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}",
        f"data:image/jpeg;base64,{generate_synthetic_image(384, 256)['base64']}",
    ]

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url_list, content_text=MULTI_EDIT_PROMPT)

    # List height/width: client sends 512×512 and 768×512 in parallel, merges to two images; assertions use lists.
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": [512, 768],
            "width": [512, 512],
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
