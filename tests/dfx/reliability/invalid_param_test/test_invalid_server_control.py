# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni sleep/wakeup JSON validation and level-2 wake contract (live ``Qwen/Qwen-Image``)."""

from __future__ import annotations

from typing import Any, Literal

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.slow, pytest.mark.diffusion, pytest.mark.omni]

_SKIP_ISSUE_3649 = pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/3649")

# One module-scoped server for this file. ``--enable-sleep-mode`` is required by the
# level-2 wake contract test; invalid-JSON cases only hit request validation and are
# unaffected. Do not mix ``omni_server`` (module) with ``omni_server_function`` here:
# ``iter_omni_server`` holds ``omni_fixture_lock`` across the module yield, so a
# function-scoped second server would deadlock / contend for the GPU.
_QWEN_IMAGE = [
    pytest.param(
        OmniServerParams(
            model="Qwen/Qwen-Image",
            server_args=["--enable-sleep-mode"],
        ),
        id="qwen_image",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# POST omni sleep / omni wakeup (invalid JSON)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "endpoint, json_body, err_message",
    [
        pytest.param("sleep", {"level": 2}, ("stage_ids", "Field required", "Missing"), id="sleep_missing_stage_ids"),
        pytest.param(
            "sleep",
            {"stage_ids": [], "level": 2},
            ("stage_ids", "Field required", "Missing"),
            id="sleep_empty_stage_ids",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param(
            "sleep",
            {"stage_ids": "not-a-list", "level": 2},
            ("stage_ids", "list_type", "a valid list"),
            id="sleep_invalid_stage_ids_type",
        ),
        pytest.param(
            "sleep",
            {"stage_ids": [0], "level": "high"},
            ("level", "int_parsing", "a valid integer"),
            id="sleep_invalid_level_type",
        ),
        pytest.param(
            "sleep",
            {"stage_ids": [0], "level": -1},
            ("level", "greater"),
            id="sleep_negative_level",
            marks=_SKIP_ISSUE_3649,
        ),
        pytest.param("wakeup", {}, ("stage_ids", "missing", "Field required"), id="wakeup_missing_stage_ids"),
        pytest.param(
            "wakeup",
            {"stage_ids": None},
            ("stage_ids", "list_type", "a valid list"),
            id="wakeup_invalid_stage_ids_type",
        ),
        pytest.param(
            "wakeup",
            {"stage_ids": []},
            ("stage_ids", "length"),
            id="wakeup_empty_stage_ids",
            marks=_SKIP_ISSUE_3649,
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN_IMAGE, indirect=True)
def test_omni_sleep_wakeup_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    endpoint: Literal["sleep", "wakeup"],
    json_body: dict[str, Any],
    err_message: str | tuple[str, ...],
) -> None:
    cfg: dict[str, Any] = {
        "json": json_body,
        "timeout": 120,
        "err_code": 400,
        "err_message": err_message,
    }
    if endpoint == "sleep":
        openai_client.send_omni_sleep_http_request(cfg)
    else:
        openai_client.send_omni_wakeup_http_request(cfg)


# Keep this after the invalid-JSON cases: sleep(level=2) discards weights and the
# shared module server is not usable afterward; module teardown then stops it.
@pytest.mark.parametrize("omni_server", _QWEN_IMAGE, indirect=True)
def test_wakeup_after_level2_sleep_fails(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Regression for #4473 Repro A via HTTP: wake after sleep(level=2) → 501.

    Moved from ``tests/entrypoints/test_omni_sleep_mode.py`` (BAGEL cold start).
    Live serve surfaces ``NotImplementedError`` as OpenAI-style JSON with code 501.
    """
    assert omni_server is not None
    sleep_resps = openai_client.send_omni_sleep_http_request(
        {
            "json": {"stage_ids": [0], "level": 2},
            "timeout": 180,
        }
    )
    sleep_resp = sleep_resps[0]
    assert sleep_resp.status_code == 200, sleep_resp
    assert sleep_resp.json_body is not None
    assert sleep_resp.json_body.get("status") == "SUCCESS", sleep_resp.json_body

    openai_client.send_omni_wakeup_http_request(
        {
            "json": {"stage_ids": [0]},
            "timeout": 120,
            "err_code": 501,
            "err_message": ("sleep(level=2)", "not yet implemented"),
        }
    )
