# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E online serving test for π0 (Pi-Zero) OpenPI websocket serving.

Boots ``vllm serve --omni --deploy-config pi0.yaml`` and drives the real OpenPI
websocket (``/v1/realtime/robot/openpi``) — the same wire path a robot uses
(handshake metadata → send observation → receive action chunk). Mirrors
``tests/e2e/online_serving/test_dreamzero_expansion.py``. Needs a GPU + a
pi0_base checkpoint; skipped in CI unless explicitly run.

The in-process LeRobot parity oracle lives separately in
``tests/diffusion/models/pi0/test_pi0_parity.py``.
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

MODEL = "lerobot/pi0_base"


test_params = [
    OmniServerParams(
        model=MODEL,
        port=8092,
        server_args=[
            "--deploy-config",
            "vllm_omni/deploy/pi0.yaml",
            "--served-model-name",
            "pi0",
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
def test_pi0_openpi_online(omni_server):
    try:
        pi0_openpi_require_dependencies()
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))

    result = pi0_openpi_run_policy_session(
        host=omni_server.host,
        port=omni_server.port,
        prompt="pick up the red block and place it in the bin",
        session_id="pi0-online-e2e",
        num_steps=2,
    )

    # Asserts every returned chunk is [50, 32] + finite, and the handshake
    # metadata matches pi0.yaml's policy_server_config.
    pi0_openpi_validate_session_result(result)

    metadata = result["metadata"]
    assert tuple(metadata["image_resolution"]) == (224, 224)
    assert metadata["needs_wrist_camera"] is True
    assert metadata["needs_session_id"] is False
    assert metadata["action_space"] == "joint_position"
