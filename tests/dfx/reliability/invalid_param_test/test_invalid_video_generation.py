# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP video job routes (live ``Wan-AI/Wan2.2-T2V-A14B-Diffusers``).

WebSocket ``/v1/video/chat/stream`` and ``/v1/realtime`` invalid-input checks live in
``test_invalid_omni_chat.py`` with the same Qwen3-Omni server module fixture (Wan cannot serve those paths).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

_WAN_T2V = [
    pytest.param(
        OmniServerParams(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
        id="wan22_t2v",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]


def _minimal_video_form_data() -> dict[str, str]:
    return {"prompt": "a cat walking on grass"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/videos (async)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "form_extra, err_message",
    [
        pytest.param(None, ("prompt", "Field required", "Missing"), id="missing_prompt"),
        pytest.param({"seconds": "0"}, ("seconds", "string_pattern_mismatch", "1"), id="seconds_zero"),
        pytest.param({"size": "bad"}, ("size", "string_pattern_mismatch"), id="size_bad"),
        pytest.param(
            {"num_inference_steps": "0"},
            ("num_inference_steps", "greater_than_equal", "1"),
            id="num_inference_steps_zero",
        ),
        pytest.param(
            {"guidance_scale": "25"}, ("guidance_scale", "less_than_equal", "20"), id="guidance_scale_above_max"
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _WAN_T2V, indirect=True)
def test_post_videos_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    form_extra: dict[str, str] | None,
    err_message: str | tuple[str, ...],
) -> None:
    data: dict[str, str] = {} if form_extra is None else {**_minimal_video_form_data(), **form_extra}
    openai_client.send_videos_create_http_request(
        {"data": data, "timeout": 120, "err_code": 400, "err_message": err_message}
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/videos/sync
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "form_extra, err_message",
    [
        pytest.param(None, ("prompt", "Field required", "Missing"), id="missing_prompt"),
        pytest.param({"seconds": "0"}, ("seconds", "string_pattern_mismatch", "1"), id="seconds_zero"),
        pytest.param({"size": "bad"}, ("size", "string_pattern_mismatch"), id="size_bad"),
        pytest.param(
            {"num_inference_steps": "201"},
            ("num_inference_steps", "less_than_equal", "200"),
            id="num_inference_steps_above_max",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _WAN_T2V, indirect=True)
def test_post_videos_sync_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    form_extra: dict[str, str] | None,
    err_message: str | tuple[str, ...],
) -> None:
    data: dict[str, str] = {} if form_extra is None else {**_minimal_video_form_data(), **form_extra}
    openai_client.send_videos_sync_http_request(
        {"data": data, "timeout": 120, "err_code": 400, "err_message": err_message}
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/videos (list)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "params, err_message",
    [
        pytest.param({"limit": -1}, ("limit", "greater_than_equal", "0"), id="invalid_limit_negative"),
        pytest.param({"limit": 101}, ("limit", "less_than_equal", "100"), id="limit_above_max"),
        pytest.param({"order": "newest"}, ("order", "literal_error", "asc", "desc"), id="invalid_order"),
    ],
)
@pytest.mark.parametrize("omni_server", _WAN_T2V, indirect=True)
def test_get_videos_list_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    params: dict[str, Any],
    err_message: str | tuple[str, ...],
) -> None:
    openai_client.send_videos_list_http_request(
        {"params": params, "timeout": 120, "err_code": 400, "err_message": err_message}
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/videos/{video_id}
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("omni_server", _WAN_T2V, indirect=True)
def test_get_video_invalid_requests(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_video_retrieve_http_request(
        {
            "video_id": "does-not-exist",
            "timeout": 120,
            "err_code": 404,
            "err_message": ("not found", "video"),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /v1/videos/{video_id}
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("omni_server", _WAN_T2V, indirect=True)
def test_delete_video_invalid_requests(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_video_delete_http_request(
        {
            "video_id": "does-not-exist",
            "timeout": 120,
            "err_code": 404,
            "err_message": ("not found", "video"),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/videos/{video_id}/content
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("omni_server", _WAN_T2V, indirect=True)
def test_get_video_content_invalid_requests(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_video_content_http_request(
        {
            "video_id": "does-not-exist",
            "timeout": 120,
            "err_code": 404,
            "err_message": ("not found", "video"),
        }
    )
