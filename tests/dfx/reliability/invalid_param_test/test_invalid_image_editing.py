# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /v1/images/edits`` multipart validation (live ``Qwen/Qwen-Image-Edit``)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

_PARAMS = [
    pytest.param(
        OmniServerParams(model="Qwen/Qwen-Image-Edit"),
        id="qwen_image_edit",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]

# Populated in ``test_images_edits_invalid_requests`` from ``omni_server.model``.
_IMAGE_EDITS_SERVER_MODEL = object()


def _finalize_edits_form_data(template: dict[str, Any], server_model: str) -> dict[str, Any]:
    return {k: (server_model if v is _IMAGE_EDITS_SERVER_MODEL else v) for k, v in template.items()}


@pytest.mark.parametrize(
    "include_image, data_template, err_message, err_code",
    [
        pytest.param(
            False,
            {"prompt": "make it brighter", "model": _IMAGE_EDITS_SERVER_MODEL},
            ("image", "required", "Unprocessable Entity"),
            422,
            id="missing_image",
        ),
        pytest.param(
            True,
            {"prompt": "edit", "model": "wrong-model-id"},
            ("model", "mismatch"),
            400,
            id="model_mismatch",
        ),
        pytest.param(
            True,
            {"prompt": "edit", "response_format": "url"},
            ("response_format", "b64_json", "supported"),
            400,
            id="unsupported_response_format",
        ),
        pytest.param(
            True,
            {"prompt": "edit", "model": _IMAGE_EDITS_SERVER_MODEL, "output_compression": "101"},
            ("less_than_equal", "output_compression", "100"),
            400,
            id="output_compression_above_max",
        ),
        pytest.param(
            True,
            {"prompt": "edit", "model": _IMAGE_EDITS_SERVER_MODEL, "layers": "1"},
            ("Invalid layers", "2", "10"),
            400,
            id="invalid_layers",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_images_edits_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    tiny_png_bytes: bytes,
    include_image: bool,
    data_template: dict[str, Any],
    err_message: str | tuple[str, ...],
    err_code: int | tuple[int, ...],
) -> None:
    cfg: dict[str, Any] = {
        "data": _finalize_edits_form_data(data_template, omni_server.model),
        "timeout": 300,
        "err_code": err_code,
        "err_message": err_message,
    }
    if include_image:
        cfg["files"] = {"image": ("x.png", tiny_png_bytes, "image/png")}
    openai_client.send_images_edits_http_request(cfg)
