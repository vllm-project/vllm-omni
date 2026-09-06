# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""E2E online serving test for π0.5 OpenPI websocket serving.

Boots ``vllm serve --omni --deploy-config pi05.yaml`` and drives the real OpenPI
websocket (``/v1/realtime/robot/openpi``) — the same wire path a robot uses
(handshake metadata → send observation → receive action chunk). Mirrors
``tests/e2e/online_serving/test_pi0_expansion.py``. Needs one H100 and the full
π0.5 checkpoint.

The ``pi0_openpi_*`` helpers are protocol-level (handshake, send obs, receive
chunk) and not π0-specific, so π0.5 reuses them; only the deploy config and the
expected metadata differ. This file is selected explicitly by the nightly H100
robot-policy job.

The in-process LeRobot parity oracle lives separately in
``tests/diffusion/models/pi05/test_pi05_parity.py``.
"""

from __future__ import annotations

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import (
    OmniServerParams,
    get_open_port,
    pi0_openpi_require_dependencies,
    pi0_openpi_run_policy_session,
    pi0_openpi_validate_session_result,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "lerobot/pi05_base"


test_params = [
    OmniServerParams(
        model=MODEL,
        port=8093,
        server_args=[
            "--deploy-config",
            "vllm_omni/deploy/pi05.yaml",
            "--served-model-name",
            "pi05",
            "--enforce-eager",
            "--disable-log-stats",
        ],
        env_dict={
            "ATTENTION_BACKEND": "torch",
            "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "MASTER_PORT": str(get_open_port()),
        },
    )
]


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_pi05_openpi_online(omni_server):
    pi0_openpi_require_dependencies()

    result = pi0_openpi_run_policy_session(
        host=omni_server.host,
        port=omni_server.port,
        prompt="pick up the red block and place it in the bin",
        session_id="pi05-online-e2e",
        num_steps=2,
        num_inference_steps=2,
    )

    # Asserts every returned chunk is [50, 32] + finite, and the handshake
    # metadata matches pi05.yaml's policy_server_config.
    pi0_openpi_validate_session_result(result)

    metadata = result["metadata"]
    assert tuple(metadata["image_resolution"]) == (224, 224)
    assert metadata["needs_wrist_camera"] is True
    assert metadata["needs_session_id"] is False
    assert metadata["action_space"] == "joint_position"
