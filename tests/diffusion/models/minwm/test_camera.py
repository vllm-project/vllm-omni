# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""minWM trajectory strings to world-to-camera matrices, in the training convention."""

import numpy as np
import pytest

from vllm_omni.diffusion.models.minwm.camera import trajectory_cameras, trajectory_viewmats

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_trajectory_starts_at_the_identity_and_dollies_along_the_camera_axis() -> None:
    viewmats = trajectory_viewmats("w*2")
    assert viewmats.shape == (3, 4, 4)
    np.testing.assert_allclose(viewmats[0], np.eye(4))
    # Two 0.08 steps forward put the world origin 0.16 behind the camera.
    np.testing.assert_allclose(viewmats[2][:3, 3], [0.0, 0.0, -0.16], atol=1e-6)


def test_moves_follow_the_heading_after_a_yaw() -> None:
    camera_to_world = np.linalg.inv(trajectory_viewmats("l*1,w*1")[-1])
    theta = np.radians(3.0)
    np.testing.assert_allclose(camera_to_world[:3, 3], 0.08 * np.array([np.sin(theta), 0.0, np.cos(theta)]), atol=1e-6)


def test_cameras_carry_one_normalized_intrinsic_per_frame() -> None:
    viewmats, intrinsics = trajectory_cameras("a*4,w*8,s*7")
    assert viewmats.shape == (20, 4, 4)
    assert intrinsics.shape == (20, 3, 3)
    np.testing.assert_allclose(intrinsics[0].numpy(), [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize(("trajectory", "message"), [("w*2,x*1", "Unknown trajectory direction"), ("w2", "segment")])
def test_malformed_trajectories_are_rejected(trajectory: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        trajectory_viewmats(trajectory)
