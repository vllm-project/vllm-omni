# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Malformed request ``attention_schedule`` values are client errors on every stage client.

A real ``vllm-omni serve`` runs ``tiny-random/Qwen-Image`` twice: the default single-stage launch
uses the inline stage client, and ``--stage-id 0`` with an Omni master runs the stage in a
``StageDiffusionProc`` subprocess. Each server must answer every malformed schedule with HTTP 400
and then still generate an image for a valid request.

From the repository root::

    pytest -s -v tests/e2e/online_serving/test_diffusion_attention_schedule_errors.py
"""

import base64
import os

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

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


def _process_stage_args() -> list[str]:
    return ["--stage-id", "0", "--omni-master-address", "127.0.0.1", "--omni-master-port", str(get_open_port())]


@pytest.fixture(scope="module", params=["inline", "process"])
def schedule_server(request: pytest.FixtureRequest):
    serve_args = ["--stage-init-timeout", "600", "--init-timeout", "900"]
    if request.param == "process":
        serve_args += _process_stage_args()
    with OmniServer(MODEL, serve_args) as server:
        yield server


def _generate(server: OmniServer, extra_params: dict | None = None) -> requests.Response:
    body = dict(BASE_REQUEST)
    if extra_params is not None:
        body["extra_params"] = extra_params
    return requests.post(
        f"http://{server.host}:{server.port}/v1/images/generations",
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
@pytest.mark.parametrize(("schedule", "message"), MALFORMED_SCHEDULES)
def test_malformed_attention_schedule_returns_400(schedule_server: OmniServer, schedule, message: str) -> None:
    response = _generate(schedule_server, {"attention_schedule": schedule})

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == 400
    assert message in error["message"]
    _assert_image(_generate(schedule_server))
