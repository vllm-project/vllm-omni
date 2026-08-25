# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E online serving tests for DreamZero OpenPI websocket serving.

Asserts the fixed OpenPI serving contract (metadata / action tensors / reset).
No upstream DreamZero reference server is required.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import (
    OmniServerParams,
    OpenPIWebSocketResponse,
    get_open_port,
)
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "GEAR-Dreams/DreamZero-DROID"
STAGE_CONFIG = get_deploy_config_path("dreamzero_tp1_cfg2.yaml")
SESSION_ID = "dreamzero-online-e2e"
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
ACTION_HORIZON = 24
ACTION_DIM = 8
CAMERA_KEYS = (
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)

# Fixed OpenPI server metadata contract for DreamZero-DROID.
EXPECTED_METADATA = {
    "image_resolution": [180, 320],
    "n_external_cameras": 2,
    "needs_wrist_camera": True,
    "needs_stereo_camera": False,
    "needs_session_id": True,
    "action_space": "joint_position",
}

# Fixed action-tensor contract for the session below:
# infer → infer → reset → infer  => 3 action tensors.
EXPECTED_NUM_ACTIONS = 3
EXPECTED_ACTION_SHAPE = (ACTION_HORIZON, ACTION_DIM)

# Child-process knobs still travel through env (server reads os.environ).
SERVER_ENV = {
    "ATTENTION_BACKEND": "torch",
    "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "MASTER_PORT": str(get_open_port()),
}


def _ensure_openpi_client_for_case() -> None:
    """PyPI openpi-client pins numpy<2; provide a local stand-in for this case only."""
    try:
        import openpi_client  # noqa: F401

        return
    except ImportError:
        pass

    from vllm_omni.entrypoints.openpi.connection import _pack, _unpack

    class _Packer:
        def pack(self, obj: Any) -> bytes:
            return _pack(obj)

    openpi_client = types.ModuleType("openpi_client")
    msgpack_numpy = types.ModuleType("openpi_client.msgpack_numpy")
    msgpack_numpy.Packer = _Packer
    msgpack_numpy.packb = _pack
    msgpack_numpy.unpackb = _unpack
    openpi_client.msgpack_numpy = msgpack_numpy
    sys.modules["openpi_client"] = openpi_client
    sys.modules["openpi_client.msgpack_numpy"] = msgpack_numpy


_ensure_openpi_client_for_case()

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG,
        env_dict=SERVER_ENV,
    )
]


def _synthetic_camera_frames() -> dict[str, np.ndarray]:
    """Build enough RGB frames in memory for a 2-chunk DreamZero schedule (indices ≤ 23)."""
    height, width, num_frames = 180, 320, 24
    camera_frames: dict[str, np.ndarray] = {}
    for channel, camera_key in enumerate(CAMERA_KEYS):
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for frame_idx in range(num_frames):
            frames[frame_idx, ..., channel] = (frame_idx * 7) % 255
            frames[frame_idx, ..., (channel + 1) % 3] = 64
        camera_frames[camera_key] = frames
    return camera_frames


def _make_observation(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for camera_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[camera_key] = selected[0] if len(frame_indices) == 1 else selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = PROMPT
    obs["session_id"] = SESSION_ID
    return obs


def _build_openpi_operations() -> list[dict[str, Any]]:
    camera_frames = _synthetic_camera_frames()
    # Chunk 0 uses the first frame; chunk 1 uses the relative offsets around frame 23.
    observations = [
        _make_observation(camera_frames, [0]),
        _make_observation(camera_frames, [0, 7, 15, 23]),
    ]
    operations = [{"endpoint": "infer", "payload": obs} for obs in observations]
    operations.append({"endpoint": "reset", "payload": {}})
    operations.append({"endpoint": "infer", "payload": observations[0]})
    return operations


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    if isinstance(normalized.get("image_resolution"), tuple):
        normalized["image_resolution"] = list(normalized["image_resolution"])
    return normalized


def _validate_dreamzero_openpi_session(response: OpenPIWebSocketResponse) -> None:
    assert _normalize_metadata(response.server_metadata) == EXPECTED_METADATA

    # Session order is infer(+)* → reset → infer; reset is not the final frame.
    reset_responses = [item for item in response.operation_responses if isinstance(item, dict) and "status" in item]
    assert reset_responses, "expected a reset response in the DreamZero OpenPI session"
    assert reset_responses[-1]["status"] == "reset successful"

    action_tensors = response.action_tensors
    assert action_tensors is not None
    assert len(action_tensors) == EXPECTED_NUM_ACTIONS
    for index, action in enumerate(action_tensors):
        assert action.shape == EXPECTED_ACTION_SHAPE, f"action[{index}] shape={action.shape}"
        assert action.dtype == np.float32, f"action[{index}] dtype={action.dtype}"
        assert np.isfinite(action).all(), f"action[{index}] has non-finite values"


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dreamzero_openpi_online(omni_server, openai_client) -> None:
    response = openai_client.send_robot_openpi_ws_request({"operations": _build_openpi_operations()})[0]
    _validate_dreamzero_openpi_session(response)
