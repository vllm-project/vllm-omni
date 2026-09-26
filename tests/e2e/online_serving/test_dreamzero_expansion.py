# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E online serving tests for DreamZero OpenPI websocket serving."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import (
    DREAMZERO_ACTION_DIM,
    DREAMZERO_ACTION_HORIZON,
    DREAMZERO_CAMERA_FILES,
    OmniServerParams,
    OpenPIWebSocketResponse,
    get_open_port,
)
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "GEAR-Dreams/DreamZero-DROID"

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=get_deploy_config_path("dreamzero_tp1_cfg2.yaml"),
        env_dict={
            "ATTENTION_BACKEND": "torch",
            "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "MASTER_PORT": str(get_open_port()),
        },
    )
]


def _write_synthetic_video(path: Path, *, channel: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, num_frames = 180, 320, 24
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., channel] = (frame_idx * 7) % 255
            frame[..., (channel + 1) % 3] = 64
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_synthetic_dreamzero_videos(video_dir: Path) -> None:
    for channel, file_name in enumerate(DREAMZERO_CAMERA_FILES.values()):
        _write_synthetic_video(video_dir / file_name, channel=channel)


def _validate_dreamzero_openpi_session(response: OpenPIWebSocketResponse) -> None:
    metadata = response.server_metadata
    assert tuple(metadata["image_resolution"]) == (180, 320)
    assert metadata["n_external_cameras"] == 2
    assert metadata["needs_wrist_camera"] is True
    assert metadata["needs_stereo_camera"] is False
    assert metadata["needs_session_id"] is True
    assert metadata["action_space"] == "joint_position"
    assert response.operation_responses[-1]["status"] == "reset successful"

    action_tensors = response.action_tensors
    assert action_tensors is not None
    assert len(action_tensors) == 3
    for index, action in enumerate(action_tensors):
        assert action.shape == (DREAMZERO_ACTION_HORIZON, DREAMZERO_ACTION_DIM)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dreamzero_openpi_online(omni_server, openai_client, tmp_path: Path) -> None:
    video_dir = tmp_path / "dreamzero_videos"
    _write_synthetic_dreamzero_videos(video_dir)
    response = openai_client.send_robot_openpi_ws_request(
        {
            "run_dreamzero_policy_session": True,
            "video_dir": video_dir,
            "session_id": "dreamzero-online-e2e",
        }
    )[0]
    _validate_dreamzero_openpi_session(response)
