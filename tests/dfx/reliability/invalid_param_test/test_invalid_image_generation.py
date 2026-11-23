# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``POST /v1/images/generations`` validation (live ``Qwen/Qwen-Image``)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

_SKIP_ISSUE_3649 = pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/3649")

_PARAMS = [
    pytest.param(
        OmniServerParams(model="Qwen/Qwen-Image"),
        id="qwen_image",
        marks=hardware_marks(res={"cuda": "H100", "rocm": "MI325"}),
    ),
]


def _minimal_images_gen_json(omni_server: OmniServer) -> dict[str, object]:
    return {"prompt": "a simple red apple icon", "model": omni_server.model}


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/images/generations (JSON)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body_spec, err_message",
    [
        pytest.param(
            {"prompt": "cat", "model": "not-the-loaded-checkpoint"},
            ("model", "mismatch"),
            id="model_mismatch",
        ),
        pytest.param({"prompt": ""}, ("prompt", "empty"), id="prompt_empty", marks=_SKIP_ISSUE_3649),
        pytest.param({"prompt": "   "}, ("prompt", "empty"), id="prompt_whitespace_only", marks=_SKIP_ISSUE_3649),
        pytest.param({"n": 0}, ("n", "greater_than_equal", "1"), id="n_below_min"),
        pytest.param({"n": 11}, ("n", "less_than_equal", "10"), id="n_above_max"),
        pytest.param({"size": "1024"}, ("size", "value_error", "WIDTHxHEIGHT"), id="size_missing_height_token"),
        pytest.param({"size": "not-a-size"}, ("size", "value_error", "WIDTHxHEIGHT"), id="size_missing_separator"),
        pytest.param({"size": "1024xabc"}, ("Invalid size format", "integers"), id="size_non_integer_height"),
        pytest.param({"size": "1024x"}, ("Invalid size format", "integers"), id="size_incomplete_height"),
        pytest.param({"size": "0x1024"}, ("Invalid size", "positive integers"), id="size_non_positive_width"),
        pytest.param(
            {"response_format": "url"},
            ("response_format", "value_error", "b64_json"),
            id="response_format_not_b64_json",
        ),
        pytest.param({"seed": -1}, ("seed", "greater_than_equal", "0"), id="seed_negative", marks=_SKIP_ISSUE_3649),
        pytest.param(
            {"seed": 2**32},
            ("seed", "less_than_equal", "4294967295"),
            id="seed_above_uint32",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            {"output_format": "gif"},
            ("output_format", "value_error", "b64_json"),
            id="output_format_invalid",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            {"lora": {"foo": "bar"}}, ("lora", "both name and path", "required"), id="lora_missing_required_fields"
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_images_generations_invalid_requests(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    body_spec: Any,
    err_message: str | tuple[str, ...],
) -> None:
    """Invalid JSON bodies for ``POST /v1/images/generations`` (model mismatch and serving-specific fields)."""
    assert isinstance(body_spec, dict)
    body = _minimal_images_gen_json(omni_server)
    body.update(body_spec)
    online_client.send_images_generations_http_request(
        {"json": body, "timeout": 300, "err_code": 400, "err_message": err_message}
    )
