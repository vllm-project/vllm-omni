# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-worker startup coverage for video output transport configuration."""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

MODEL = os.environ.get(
    "VLLM_OMNI_VIDEO_TRANSPORT_STARTUP_MODEL",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
)

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.gpu]

_BASE_SERVER_ARGS = [
    "--trust-remote-code",
    "--num-gpus",
    "2",
    "--tensor-parallel-size",
    "2",
    "--enforce-eager",
]

_STARTUP_CASES = [
    pytest.param(OmniServerParams(model=MODEL, server_args=_BASE_SERVER_ARGS), id="omitted"),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                *_BASE_SERVER_ARGS,
                "--video-output-transport",
                '{"enable_device_postprocess": false}',
            ],
        ),
        id="disabled",
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                *_BASE_SERVER_ARGS,
                "--video-output-transport",
                '{"enable_device_postprocess": true}',
            ],
        ),
        id="enabled",
    ),
]


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _STARTUP_CASES, indirect=True)
def test_video_transport_config_reaches_two_worker_readiness(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    responses = openai_client.send_health_http_request({"timeout": 10})

    assert responses[0].status_code == 200
    assert omni_server.proc is not None
    assert omni_server.proc.poll() is None
