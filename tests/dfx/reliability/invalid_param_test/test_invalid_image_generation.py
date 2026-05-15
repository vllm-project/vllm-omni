# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /v1/images/generations`` validation (live ``Qwen/Qwen-Image``)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

_SKIP_ISSUE_3649 = pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/3649")

_PARAMS = [
    pytest.param(
        OmniServerParams(model="Qwen/Qwen-Image"),
        id="qwen_image",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]

# Full replacement body for ``test_images_generations_invalid_requests`` (no minimal prompt/model merge).
_IMAGES_GEN_MISSING_PROMPT_JSON = object()


def _minimal_images_gen_json(omni_server: OmniServer) -> dict[str, object]:
    return {"prompt": "a simple red apple icon", "model": omni_server.model}


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/images/generations (JSON)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body_spec, err_message",
    [
        pytest.param(_IMAGES_GEN_MISSING_PROMPT_JSON, ("prompt", "Field required", "Missing"), id="missing_prompt"),
        pytest.param(
            {"prompt": "cat", "model": "not-the-loaded-checkpoint"},
            ("model", "mismatch"),
            id="model_mismatch",
        ),
        pytest.param({"prompt": ""}, ("prompt", "empty"), id="prompt_empty", marks=_SKIP_ISSUE_3649),
        pytest.param({"prompt": "   "}, ("prompt", "empty"), id="prompt_whitespace_only", marks=_SKIP_ISSUE_3649),
        pytest.param({"prompt": 123}, ("prompt", "string_type", "valid string"), id="prompt_wrong_type"),
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
        pytest.param(
            {"use_system_prompt": "bogus"},
            ("use_system_prompt", "value_error", "dynamic"),
            id="use_system_prompt_invalid",
        ),
        pytest.param(
            {"num_inference_steps": 0},
            ("num_inference_steps", "'greater_than_equal", "1"),
            id="num_inference_steps_below_min",
        ),
        pytest.param(
            {"num_inference_steps": 201},
            ("num_inference_steps", "less_than_equal", "200"),
            id="num_inference_steps_above_max",
        ),
        pytest.param(
            {"guidance_scale": -1.0}, ("guidance_scale", "greater_than_equal", "0"), id="guidance_scale_negative"
        ),
        pytest.param(
            {"guidance_scale": 21.0}, ("guidance_scale", "less_than_equal", "20"), id="guidance_scale_above_max"
        ),
        pytest.param(
            {"true_cfg_scale": -1.0}, ("true_cfg_scale", "greater_than_equal", "0"), id="true_cfg_scale_negative"
        ),
        pytest.param(
            {"true_cfg_scale": 21.0}, ("true_cfg_scale", "less_than_equal", "20"), id="true_cfg_scale_above_max"
        ),
        pytest.param({"layers": 2}, ("layers", "value_error", "3", "10"), id="layers_below_min"),
        pytest.param({"layers": 11}, ("layers", "value_error", "3", "10"), id="layers_above_max"),
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
            {"vae_use_slicing": "wrong_type"},
            ("vae_use_slicing", "bool_parsing", "validation error"),
            id="vae_use_slicing_wrong_type",
        ),
        pytest.param({"user": 123}, ("user", "string_type", "valid string"), id="user_wrong_type"),
        pytest.param(
            {"negative_prompt": ["noise"]},
            ("negative_prompt", "string_type", "valid string"),
            id="negative_prompt_wrong_type",
        ),
        pytest.param(
            {"generator_device": 1},
            ("generator_device", "string_type", "valid string"),
            id="generator_device_wrong_type",
        ),
        pytest.param(
            {"lora": {"foo": "bar"}}, ("lora", "both name and path", "required"), id="lora_missing_required_fields"
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_images_generations_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    body_spec: Any,
    err_message: str | tuple[str, ...],
) -> None:
    """Invalid JSON bodies for ``POST /v1/images/generations`` (missing prompt, model mismatch, bad fields)."""
    if body_spec is _IMAGES_GEN_MISSING_PROMPT_JSON:
        body: dict[str, object] = {"size": "1024x1024"}
    else:
        assert isinstance(body_spec, dict)
        body = _minimal_images_gen_json(omni_server)
        body.update(body_spec)
    openai_client.send_images_generations_http_request(
        {"json": body, "timeout": 300, "err_code": 400, "err_message": err_message}
    )
