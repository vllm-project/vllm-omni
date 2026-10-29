# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end online serving test for CAE-FastWAM through the OpenPI robot endpoint."""

import os

import numpy as np
import pytest

from tests.diffusion.models.fastwam import reference_observation as ref
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, OpenPIWebSocketResponse
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "amd/CAE-FastWAM"

pytest.importorskip("websockets")
pytest.importorskip("openpi_client.msgpack_numpy")

test_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("cae_fastwam_libero.yaml"),
            env_dict={"VLLM_DISABLE_COMPILE_CACHE": "1"},
            init_timeout=1200,
            stage_init_timeout=900,
        ),
        id="cae-fastwam-openpi",
    )
]


def _session(openai_client, observation: dict) -> OpenPIWebSocketResponse:
    return openai_client.send_robot_openpi_ws_request(
        {
            "operations": [
                {"endpoint": "reset", "payload": {}},
                {"endpoint": "infer", "payload": observation},
            ],
        }
    )[0]


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_cae_fastwam_openpi_online(omni_server, openai_client) -> None:
    response = _session(openai_client, ref.build_libero_reference_observation(seed=None))
    metadata = response.server_metadata
    assert metadata["action_horizon"] == 32 and metadata["action_dim"] == 7
    assert metadata["needs_wrist_camera"] is True and metadata["needs_session_id"] is False
    assert list(metadata["image_resolution"]) == [224, 224]

    actions = response.actions
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (32, 7) and actions.dtype == np.float32
    assert np.isfinite(actions).all()
    assert set(np.unique(actions[:, -1])) <= {-1.0, 1.0}  # binarised LIBERO gripper


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_cae_fastwam_openpi_precision(omni_server, openai_client) -> None:
    """Actions match the original CAE-FastWAM eager path (request seed=42)."""
    actions = _session(openai_client, ref.build_libero_reference_observation(seed=ref.SEED)).actions
    assert actions is not None
    rows = actions[list(ref.REFERENCE_ROWS)]
    np.testing.assert_allclose(
        rows[:, :6],
        ref.REFERENCE_CHUNK_ROWS[:, :6],
        atol=ref.GPU_ATOL,
        rtol=0.0,
        err_msg="CAE-FastWAM action mismatch vs the reference implementation",
    )
    np.testing.assert_array_equal(rows[:, 6], ref.REFERENCE_CHUNK_ROWS[:, 6])
