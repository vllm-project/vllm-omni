# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Malformed request ``attention_schedule`` and ``attention_sigma_schedule`` values return HTTP 400
on the image and chat routes.

A real ``vllm-omni serve`` runs ``tiny-random/Qwen-Image`` twice: the default single-stage launch
uses the inline stage client, and ``--stage-id 0`` with an Omni master runs the stage in a
``StageDiffusionProc`` subprocess. The routes reject a malformed schedule before dispatch; an
unknown profile passes them and is rejected at scheduler admission inside the stage, so its client
error must come back through the stage client. After each rejection the server must still generate
an image for a valid request.

From the repository root::

    pytest -s -v tests/e2e/online_serving/test_diffusion_attention_schedule_errors.py
"""

import base64
import os

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "tiny-random/Qwen-Image"
REQUEST_TIMEOUT_S = 600
BASE_REQUEST = {"prompt": "a cat", "size": "64x64", "num_inference_steps": 2, "seed": 1}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MALFORMED_SCHEDULES = [
    pytest.param({}, "must be a list", id="object-instead-of-list"),
    pytest.param("dense", "must be a list", id="string-instead-of-list"),
    pytest.param(
        [{"start": 0, "end": 4, "profile": "a"}, {"start": 2, "end": None, "profile": "b"}],
        "must not overlap",
        id="overlapping-ranges",
    ),
    pytest.param(
        [{"start": 2, "end": None, "profile": "a"}, {"start": 0, "end": 1, "profile": "b"}],
        "must not overlap",
        id="unordered-ranges",
    ),
    pytest.param([{"start": 0, "profile": "a"}], "exactly start, end and profile", id="missing-end"),
    pytest.param([{"start": 3, "end": 3, "profile": "a"}], "greater than start", id="empty-range"),
    pytest.param([{"start": True, "end": None, "profile": "a"}], "must be an integer", id="bool-step"),
    pytest.param([{"start": 0, "end": None, "profile": "1bad"}], "profile name", id="invalid-profile-name"),
    pytest.param([{"start": 0, "end": None, "profile": "dense"}], "unknown profile", id="unknown-profile"),
]

MALFORMED_SIGMA_SCHEDULES = [
    pytest.param({}, "must be a list", id="object-instead-of-list"),
    pytest.param("dense", "must be a list", id="string-instead-of-list"),
    pytest.param(
        [{"low": 0.0, "high": 0.6, "profile": "a"}, {"low": 0.5, "high": 1.0, "profile": "b"}],
        "must not overlap",
        id="overlapping-windows",
    ),
    pytest.param(
        [{"low": 0.5, "high": 1.0, "profile": "a"}, {"low": 0.0, "high": 0.5, "profile": "b"}],
        "ascending low",
        id="descending-windows",
    ),
    pytest.param([{"low": 0.0, "profile": "a"}], "exactly low, high and profile", id="missing-high"),
    pytest.param([{"low": 0.5, "high": 0.5, "profile": "a"}], "greater than low", id="empty-window"),
    pytest.param([{"low": 0.0, "high": 1.5, "profile": "a"}], "must be in [0, 1]", id="bound-above-one"),
    pytest.param([{"low": 0, "high": 10**400, "profile": "a"}], "must be in [0, 1]", id="oversized-int-bound"),
    pytest.param([{"low": True, "high": 1.0, "profile": "a"}], "must be a number", id="bool-bound"),
    pytest.param([{"low": 0.0, "high": 1.0, "profile": "1bad"}], "profile name", id="invalid-profile-name"),
    pytest.param([{"low": 0.0, "high": 1.0, "profile": "dense"}], "unknown profile", id="unknown-profile"),
]

# For each schedule kind, one schedule the chat route rejects itself and one that only scheduler admission rejects.
MALFORMED_CHAT_SCHEDULES = [MALFORMED_SCHEDULES[1], MALFORMED_SCHEDULES[-1]]
MALFORMED_CHAT_SIGMA_SCHEDULES = [MALFORMED_SIGMA_SCHEDULES[1], MALFORMED_SIGMA_SCHEDULES[-1]]


SERVERS = [
    pytest.param(OmniServerParams(model=MODEL), id="inline"),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--stage-id",
                "0",
                "--omni-master-address",
                "127.0.0.1",
                "--omni-master-port",
                str(get_open_port()),
            ],
        ),
        id="process",
    ),
]


def _generate(server: OmniServer, extra_params: dict | None = None) -> requests.Response:
    body = dict(BASE_REQUEST)
    if extra_params is not None:
        body["extra_params"] = extra_params
    return requests.post(
        f"http://{server.host}:{server.port}/v1/images/generations",
        json=body,
        timeout=REQUEST_TIMEOUT_S,
    )


def _chat(server: OmniServer, extra_args: dict[str, object], *, in_extra_body: bool) -> requests.Response:
    """Send ``extra_args`` at the request root or inside ``extra_body``."""
    extra_body: dict[str, object] = {"height": 64, "width": 64, "num_inference_steps": 2, "seed": 1}
    body: dict[str, object] = {"messages": [{"role": "user", "content": "a cat"}], "extra_body": extra_body}
    (extra_body if in_extra_body else body)["extra_args"] = extra_args
    return requests.post(
        f"http://{server.host}:{server.port}/v1/chat/completions",
        json=body,
        timeout=REQUEST_TIMEOUT_S,
    )


def _assert_image(response: requests.Response) -> None:
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 1
    assert base64.b64decode(data[0]["b64_json"]).startswith(PNG_MAGIC)


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
@pytest.mark.parametrize(("schedule", "message"), MALFORMED_SCHEDULES)
def test_malformed_attention_schedule_returns_400(omni_server: OmniServer, schedule, message: str) -> None:
    response = _generate(omni_server, {"attention_schedule": schedule})

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == 400
    assert message in error["message"]
    _assert_image(_generate(omni_server))


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
@pytest.mark.parametrize(("schedule", "message"), MALFORMED_SIGMA_SCHEDULES)
def test_malformed_attention_sigma_schedule_returns_400(omni_server: OmniServer, schedule, message: str) -> None:
    response = _generate(omni_server, {"attention_sigma_schedule": schedule})

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == 400
    assert message in error["message"]
    _assert_image(_generate(omni_server))


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
@pytest.mark.parametrize("in_extra_body", [False, True], ids=["root-extra-args", "extra-body-extra-args"])
@pytest.mark.parametrize(("schedule", "message"), MALFORMED_CHAT_SCHEDULES)
def test_malformed_chat_attention_schedule_returns_400(
    omni_server: OmniServer, in_extra_body: bool, schedule, message: str
) -> None:
    response = _chat(omni_server, {"attention_schedule": schedule}, in_extra_body=in_extra_body)

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == 400
    assert message in error["message"]
    _assert_image(_generate(omni_server))


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
@pytest.mark.parametrize("in_extra_body", [False, True], ids=["root-extra-args", "extra-body-extra-args"])
@pytest.mark.parametrize(("schedule", "message"), MALFORMED_CHAT_SIGMA_SCHEDULES)
def test_malformed_chat_attention_sigma_schedule_returns_400(
    omni_server: OmniServer, in_extra_body: bool, schedule, message: str
) -> None:
    response = _chat(omni_server, {"attention_sigma_schedule": schedule}, in_extra_body=in_extra_body)

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == 400
    assert message in error["message"]
    _assert_image(_generate(omni_server))
