# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world.actions import (
    camera_trajectory_from_absolute_pose,
    integrate_lingbot_camera_actions,
    parse_lingbot_camera_action_script,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_action_integrator_matches_lingbot_motion_and_calibration() -> None:
    trajectory, pitch = integrate_lingbot_camera_actions(
        [["w"], ["d"], ["l"]],
        width=832,
        height=480,
    )

    assert pitch == 0.0
    torch.testing.assert_close(
        trajectory.poses[0, :3, 3],
        torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        trajectory.poses[1, :3, 3],
        torch.tensor([0.05, 0.0, 0.05], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )
    expected_yaw = torch.tensor(
        [
            [0.9945219, 0.0, 0.10452846],
            [0.0, 1.0, 0.0],
            [-0.10452846, 0.0, 0.9945219],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        trajectory.poses[2, :3, :3],
        expected_yaw,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        trajectory.intrinsics[0],
        torch.tensor([500.0, 500.0, 416.0, 240.0]),
    )

    resized, _ = integrate_lingbot_camera_actions(
        [[]],
        width=416,
        height=240,
    )
    scaled = resized.intrinsics[0] * torch.tensor([416 / 832, 240 / 480, 416 / 832, 240 / 480])
    torch.testing.assert_close(
        scaled,
        torch.tensor([500.0, 500.0, 208.0, 120.0]),
    )

    with pytest.raises(ValueError, match="resolution must be positive"):
        camera_trajectory_from_absolute_pose(trajectory.poses[:1], width=0, height=480)
    with pytest.raises(ValueError, match=r"\[frames, 4, 4\]"):
        camera_trajectory_from_absolute_pose(torch.zeros(4, 4), width=832, height=480)


def test_parse_camera_action_script_rejects_wrong_chunk_width() -> None:
    with pytest.raises(ValueError, match="exactly 3 per-latent-frame"):
        parse_lingbot_camera_action_script([[["w"], ["w"]]], frames_per_chunk=3)
