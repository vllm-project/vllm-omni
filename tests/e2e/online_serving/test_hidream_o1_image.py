# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online serving tests for HiDream-O1-Image (text-to-image, image editing, personalization).

- ``test_t2i_dev_001``: ``HiDream-ai/HiDream-O1-Image-Dev`` — baseline text-to-image smoke.
  Carries both ``core_model`` and ``advanced_model`` marks so it runs in both L2 (test-ready)
  and L3 (test-merge) pipelines.
- ``test_image_edit_001``: single-reference image editing (``advanced_model`` only).
- ``test_multi_ref_personalization_001``: two-reference personalization (``advanced_model`` only).

From ``tests/``::

    pytest -s -v e2e/online_serving/test_hidream_o1_image.py -m "core_model and diffusion" --run-level=core_model
    pytest -s -v e2e/online_serving/test_hidream_o1_image.py -m "advanced_model and diffusion" --run-level=advanced_model
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

DEV_MODEL = "HiDream-ai/HiDream-O1-Image-Dev"

T2I_PROMPT = "A golden retriever running through a field of sunflowers at golden hour."
EDIT_PROMPT = "Make the background a snowy mountain landscape, keep the subject unchanged."
PERSONALIZATION_PROMPT = "Two friends sitting together at a cozy café in Paris."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_feature_cases(model: str):
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
@pytest.mark.parametrize("omni_server", _get_feature_cases(DEV_MODEL), indirect=True)
def test_t2i_dev_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Text-to-image baseline smoke for HiDream-O1-Image-Dev (L2 + L3)."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_feature_cases(DEV_MODEL), indirect=True)
def test_image_edit_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Single-reference image editing smoke for HiDream-O1-Image-Dev (L3)."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(256, 256)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_feature_cases(DEV_MODEL), indirect=True)
def test_multi_ref_personalization_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Two-reference personalization smoke for HiDream-O1-Image-Dev (L3)."""
    image_data_url_list = [
        f"data:image/jpeg;base64,{generate_synthetic_image(256, 256)['base64']}"
        for _ in range(2)
    ]
    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url_list, content_text=PERSONALIZATION_PROMPT
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)
