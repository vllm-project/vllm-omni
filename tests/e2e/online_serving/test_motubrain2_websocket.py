# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run a released RoboTwin or Astribot policy through its Motubrain WebSocket.

Set MOTUBRAIN2_TEST_MODEL to the prepared serving bundle. Optionally set
MOTUBRAIN2_TEST_DEPLOY_CONFIG to the matching deployment with local encoders.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from websockets.sync.client import connect

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.motubrain2_codec import pack_message, unpack_message

MODEL = os.environ.get("MOTUBRAIN2_TEST_MODEL", "")
DEPLOY = os.environ.get("MOTUBRAIN2_TEST_DEPLOY_CONFIG") or get_deploy_config_path("motubrain2.yaml")


@pytest.mark.local_model
@pytest.mark.diffusion
@pytest.mark.skipif(not MODEL, reason="Set MOTUBRAIN2_TEST_MODEL to a prepared full checkpoint")
@hardware_test(res={"cuda": "B200"})
@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY,
            server_args=["--enforce-eager", "--disable-log-stats"],
            env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            init_timeout=1200,
            stage_init_timeout=900,
        )
    ],
    indirect=True,
)
def test_motubrain2_online(omni_server):
    profile = json.loads((Path(MODEL) / "profile.json").read_text())
    cameras = [view.get("source", view["name"]) for view in profile["views"]]
    state: list[float] = []
    quaternion_offsets = []
    for field in profile["action"]["fields"]:
        if field["kind"].startswith("eef_"):
            quaternion_offsets.append(len(state) + 3)
            state.extend([0, 0, 0, 0, 0, 0, 1])
        else:
            state.extend([0] * field["width"])
    observation = {
        "prompt": "Pick up the object",
        "state": np.array(state, dtype=np.float32),
        "camera_extrinsic": np.eye(4, dtype=np.float32),
        "seed": 42,
        **{name: np.full((256, 448, 3), i * 40, dtype=np.uint8) for i, name in enumerate(cameras)},
    }
    uri = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime/robot/motubrain2"
    with connect(uri, max_size=None) as socket:
        meta = unpack_message(socket.recv(timeout=30))
        socket.send(pack_message(observation))
        actions = unpack_message(socket.recv(timeout=300))
        socket.send(pack_message({"endpoint": "reset"}))
        assert unpack_message(socket.recv(timeout=30)) == {"status": "reset successful"}
    assert meta["camera_sources"] == cameras
    assert meta["action_horizon"] == profile["action_steps"]
    assert meta["action_dim"] == len(state)
    assert (
        meta["action_space"]
        == {16: "cartesian_pose_xyzw_gripper", 25: "cartesian_pose_xyzw_joint_position_gripper"}[len(state)]
    )
    assert actions.dtype == np.float32
    assert actions.shape == (profile["action_steps"], len(state))
    assert np.isfinite(actions).all()
    for start in quaternion_offsets:
        np.testing.assert_allclose(np.linalg.norm(actions[:, start : start + 4], axis=1), 1.0, atol=1e-4)
