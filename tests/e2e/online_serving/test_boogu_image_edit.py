# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online serving tests for Boogu-Image-0.1-Edit (image-to-image via chat completions).

The Edit checkpoint shares the ``BooguImagePipeline`` class with the Base
text-to-image checkpoint; the editing (TI2I) path activates when the request
carries a reference image. Text guidance rides on ``guidance_scale`` (upstream
default 4.0), and image guidance rides on ``guidance_scale_2`` (upstream default
1.0 = off; a value > 1 enables the double-guidance path). Only a single
reference image is supported.

- ``test_single_image_to_image_001``: one reference image, text guidance only.
- ``test_double_guidance_001``: one reference image, text + image guidance.
- ``test_images_edits_endpoint_001``: same text-guided edit via ``/v1/images/edits``.
- ``test_images_edits_endpoint_double_guidance_001``: double guidance via ``/v1/images/edits``.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_boogu_image_edit.py \
        -m "full_model and diffusion and H100" --run-level=full_model
"""

import base64

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

EDIT_PROMPT = "Change the style to a colored pencil drawing."
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


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Boogu/Boogu-Image-0.1-Edit"),
    indirect=True,
)
def test_single_image_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Single-reference text-guided edit smoke for ``Boogu/Boogu-Image-0.1-Edit``."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            # Boogu text guidance (upstream default 4.0); > 1 enables CFG.
            "guidance_scale": 5.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Boogu/Boogu-Image-0.1-Edit"),
    indirect=True,
)
def test_double_guidance_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Single-reference double-guidance edit (text + image guidance) smoke."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Image guidance > 1 activates the 3-prediction double-guidance path.
            "guidance_scale_2": 2.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


def _edits_request_config(model: str, *, guidance_scale_2: float | None = None) -> dict:
    """Multipart ``/v1/images/edits`` config: one synthetic reference + smoke-depth params."""
    image_bytes = base64.b64decode(generate_synthetic_image(512, 512)["base64"])
    data = {
        "model": model,
        "prompt": EDIT_PROMPT,
        "num_inference_steps": 2,
        # Boogu text guidance (upstream default 4.0); > 1 enables CFG.
        "guidance_scale": 5.0,
        "seed": 42,
    }
    if guidance_scale_2 is not None:
        data["guidance_scale_2"] = guidance_scale_2
    return {
        "data": data,
        "files": {"image": ("ref.jpg", image_bytes, "image/jpeg")},
        "timeout": 300,
    }


def _assert_edits_response_has_image(responses) -> None:
    assert responses and responses[0].success, "Expected a successful /v1/images/edits response"
    body = responses[0].json_body
    assert body and body.get("data"), "No image data returned by /v1/images/edits"
    assert body["data"][0].get("b64_json"), "Missing b64_json in /v1/images/edits response"


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Boogu/Boogu-Image-0.1-Edit"),
    indirect=True,
)
def test_images_edits_endpoint_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Single-reference text-guided edit via the OpenAI-compatible ``/v1/images/edits`` endpoint."""
    responses = openai_client.send_images_edits_http_request(_edits_request_config(omni_server.model))
    _assert_edits_response_has_image(responses)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Boogu/Boogu-Image-0.1-Edit"),
    indirect=True,
)
def test_images_edits_endpoint_double_guidance_001(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Double-guidance edit (text + image guidance) via ``/v1/images/edits``."""
    responses = openai_client.send_images_edits_http_request(
        _edits_request_config(omni_server.model, guidance_scale_2=2.0)
    )
    _assert_edits_response_has_image(responses)
