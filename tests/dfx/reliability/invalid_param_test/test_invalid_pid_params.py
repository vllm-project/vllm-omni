# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weekly: invalid ``pid_decode`` params on Qwen-Image return 4xx.

Reuses the ``invalid_param_test`` base: real server + ``send_images_generations_http_request``.
Exact ``err_code``/``err_message`` may need a one-time calibration run before being frozen.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

MODEL = "Qwen/Qwen-Image"

_PID_SERVER = [
    pytest.param(
        OmniServerParams(
            model=MODEL, server_args=["--pid-enable", "--pid-gemma", "Efficient-Large-Model/gemma-2-2b-it"]
        ),
        id="pid_enabled",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]
_NO_PID_SERVER = [
    pytest.param(
        OmniServerParams(model=MODEL),
        id="pid_not_enabled",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]


def _minimal_body(omni_server) -> dict[str, Any]:
    return {
        "prompt": "a simple red apple icon",
        "model": omni_server.model,
        "size": "512x512",
    }


@pytest.mark.parametrize(
    "pid_decode, err_code, err_message",
    [
        pytest.param("bad", (400, 422), ("pid_decode",), id="wrong_type"),
        pytest.param({"extra": 1}, (400, 500), ("pid_decode", "extra"), id="unknown_key"),
    ],
)
@pytest.mark.parametrize("omni_server", _PID_SERVER, indirect=True)
def test_images_generations_invalid_pid_decode(omni_server, openai_client, pid_decode, err_code, err_message) -> None:
    """Malformed ``pid_decode`` values on a server started with ``--pid-enable`` -> 4xx/5xx."""
    body = _minimal_body(omni_server)
    body["pid_decode"] = pid_decode
    openai_client.send_images_generations_http_request(
        {"json": body, "timeout": 300, "err_code": err_code, "err_message": err_message}
    )


@pytest.mark.parametrize("omni_server_function", _NO_PID_SERVER, indirect=True)
def test_images_generations_pid_enabled_without_server_flag(omni_server_function, openai_client_function) -> None:
    """``pid_decode.enabled=True`` on a server without ``--pid-enable`` -> 4xx/5xx (mixin RuntimeError)."""
    body = _minimal_body(omni_server_function)
    body["pid_decode"] = {"enabled": True, "scale": 4}
    openai_client_function.send_images_generations_http_request(
        {
            "json": body,
            "timeout": 300,
            "err_code": (400, 500),
            "err_message": "--pid-enable",
        }
    )
