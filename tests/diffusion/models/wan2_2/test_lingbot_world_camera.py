# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MODULE_PATH = Path(__file__).parents[4] / "vllm_omni/diffusion/models/wan2_2/lingbot_world_camera.py"
_SPEC = importlib.util.spec_from_file_location("_lingbot_world_camera_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CAMERA_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CAMERA_MODULE
_SPEC.loader.exec_module(_CAMERA_MODULE)

CameraTrajectory = _CAMERA_MODULE.CameraTrajectory
build_plucker_embedding = _CAMERA_MODULE.build_plucker_embedding
interpolate_camera_trajectory = _CAMERA_MODULE.interpolate_camera_trajectory
load_camera_trajectory = _CAMERA_MODULE.load_camera_trajectory

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _identity_poses(num_frames: int) -> np.ndarray:
    return np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)


def _write_trajectory(
    directory: Path,
    poses: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    np.save(directory / "poses.npy", poses)
    np.save(directory / "intrinsics.npy", intrinsics)


def _trajectory(
    poses: torch.Tensor,
    intrinsics: torch.Tensor | None = None,
) -> CameraTrajectory:
    if intrinsics is None:
        intrinsics = torch.tensor([[832.0 / 3.0, 160.0, 832.0 / 3.0, 160.0]]).repeat(poses.shape[0], 1)
    return CameraTrajectory(poses=poses, intrinsics=intrinsics)


@pytest.mark.parametrize("present_file", [None, "poses.npy"])
def test_load_camera_trajectory_rejects_missing_files(tmp_path: Path, present_file: str | None) -> None:
    if present_file == "poses.npy":
        np.save(tmp_path / present_file, _identity_poses(1))

    missing_file = "poses.npy" if present_file is None else "intrinsics.npy"
    with pytest.raises(FileNotFoundError, match=missing_file):
        load_camera_trajectory(tmp_path)


@pytest.mark.parametrize(
    ("poses", "intrinsics", "message"),
    [
        (np.zeros((2, 3, 4)), np.zeros((2, 4)), "poses.npy"),
        (_identity_poses(2), np.zeros((2, 3)), "intrinsics.npy"),
        (_identity_poses(0), np.zeros((0, 4)), "at least one"),
        (_identity_poses(2), np.zeros((1, 4)), "same number of frames"),
    ],
)
def test_load_camera_trajectory_rejects_invalid_shapes(
    tmp_path: Path,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    message: str,
) -> None:
    _write_trajectory(tmp_path, poses, intrinsics)

    with pytest.raises(ValueError, match=message):
        load_camera_trajectory(tmp_path)


@pytest.mark.parametrize("array_name", ["poses", "intrinsics"])
def test_load_camera_trajectory_rejects_non_finite_values(tmp_path: Path, array_name: str) -> None:
    poses = _identity_poses(1)
    intrinsics = np.ones((1, 4), dtype=np.float64)
    if array_name == "poses":
        poses[0, 0, 0] = np.nan
    else:
        intrinsics[0, 0] = np.inf
    _write_trajectory(tmp_path, poses, intrinsics)

    with pytest.raises(ValueError, match="finite"):
        load_camera_trajectory(tmp_path)


def test_load_camera_trajectory_normalizes_c2w_poses_to_first_camera(tmp_path: Path) -> None:
    first_c2w = np.array(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    relative_pose = np.eye(4)
    relative_pose[:3, 3] = [1.0, 2.0, 3.0]
    poses = np.stack((first_c2w, first_c2w @ relative_pose))
    intrinsics = np.array([[100.0, 110.0, 120.0, 130.0]] * 2)
    _write_trajectory(tmp_path, poses, intrinsics)

    trajectory = load_camera_trajectory(tmp_path)

    assert trajectory.poses.dtype == torch.float32
    assert trajectory.intrinsics.dtype == torch.float32
    torch.testing.assert_close(trajectory.poses[0], torch.eye(4))
    torch.testing.assert_close(trajectory.poses[1], torch.from_numpy(relative_pose).float())
    torch.testing.assert_close(trajectory.intrinsics, torch.from_numpy(intrinsics).float())


def test_load_camera_trajectory_normalizes_non_commuting_rotations(tmp_path: Path) -> None:
    first_c2w = np.array(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    current_c2w = np.array(
        [
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 1.0, 0.0, 4.0],
            [-1.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert not np.allclose(first_c2w[:3, :3] @ current_c2w[:3, :3], current_c2w[:3, :3] @ first_c2w[:3, :3])
    _write_trajectory(
        tmp_path,
        np.stack((first_c2w, current_c2w)),
        np.ones((2, 4), dtype=np.float64),
    )

    trajectory = load_camera_trajectory(tmp_path)

    expected_relative_pose = np.linalg.inv(first_c2w) @ current_c2w
    torch.testing.assert_close(trajectory.poses[1], torch.from_numpy(expected_relative_pose).float())


def test_interpolate_camera_trajectory_uses_linear_and_spherical_interpolation() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    poses[1, :3, 3] = torch.tensor([2.0, 4.0, 6.0])
    intrinsics = torch.tensor([[10.0, 20.0, 30.0, 40.0], [20.0, 40.0, 60.0, 80.0]])
    trajectory = CameraTrajectory(poses=poses, intrinsics=intrinsics)

    result = interpolate_camera_trajectory(trajectory, 3)

    torch.testing.assert_close(result.poses[0], poses[0])
    torch.testing.assert_close(result.poses[-1], poses[-1])
    torch.testing.assert_close(result.poses[1, :3, 3], torch.tensor([1.0, 2.0, 3.0]))
    expected_mid_rotation = torch.tensor(
        [
            [math.sqrt(0.5), -math.sqrt(0.5), 0.0],
            [math.sqrt(0.5), math.sqrt(0.5), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(result.poses[1, :3, :3], expected_mid_rotation)
    torch.testing.assert_close(result.intrinsics[1], torch.tensor([15.0, 30.0, 45.0, 60.0]))


def test_build_plucker_embedding_has_forward_center_ray() -> None:
    trajectory = _trajectory(torch.eye(4).unsqueeze(0))

    result = build_plucker_embedding(
        trajectory,
        height=3,
        width=3,
        target_height=3,
        target_width=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(result[0, :3, 1, 1], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(result[0, 3:, 1, 1], torch.zeros(3))


def test_build_plucker_embedding_scales_reference_intrinsics() -> None:
    trajectory = _trajectory(
        torch.eye(4).unsqueeze(0),
        torch.tensor([[416.0, 240.0, 416.0, 240.0]]),
    )

    result = build_plucker_embedding(
        trajectory,
        height=4,
        width=8,
        target_height=4,
        target_width=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_corner_direction = torch.tensor([-1.0, -1.0, 1.0]) / math.sqrt(3.0)
    torch.testing.assert_close(result[0, :3, 0, 0], expected_corner_direction)


def test_build_plucker_embedding_samples_target_pixel_centers() -> None:
    trajectory = _trajectory(
        torch.eye(4).unsqueeze(0),
        torch.tensor([[416.0, 240.0, 416.0, 240.0]]),
    )

    result = build_plucker_embedding(
        trajectory,
        height=6,
        width=10,
        target_height=2,
        target_width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # The first target cell is centered at source pixel (x=0.5, y=1.0).
    expected_direction = torch.tensor([-0.9, -2.0 / 3.0, 1.0])
    expected_direction /= torch.linalg.vector_norm(expected_direction)
    torch.testing.assert_close(result[0, :3, 0, 0], expected_direction)


def test_build_plucker_embedding_transforms_rays_with_c2w_pose() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    poses[1, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    trajectory = _trajectory(poses)

    result = build_plucker_embedding(
        trajectory,
        height=3,
        width=3,
        target_height=3,
        target_width=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_direction = torch.tensor([0.0, 1.0, 1.0]) / math.sqrt(2.0)
    expected_moment = torch.tensor([0.0, -1.0, 1.0]) / math.sqrt(2.0)
    torch.testing.assert_close(result[1, :3, 1, 2], expected_direction)
    torch.testing.assert_close(result[1, 3:, 1, 2], expected_moment)


def test_build_plucker_embedding_honors_shape_dtype_and_device() -> None:
    trajectory = _trajectory(torch.eye(4).repeat(2, 1, 1))

    result = build_plucker_embedding(
        trajectory,
        height=12,
        width=20,
        target_height=3,
        target_width=5,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert result.shape == (2, 6, 3, 5)
    assert result.dtype == torch.float64
    assert result.device == torch.device("cpu")


def test_build_plucker_embedding_is_deterministic() -> None:
    trajectory = _trajectory(torch.eye(4).repeat(2, 1, 1))
    kwargs = {
        "height": 12,
        "width": 20,
        "target_height": 3,
        "target_width": 5,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }

    first = build_plucker_embedding(trajectory, **kwargs)
    second = build_plucker_embedding(trajectory, **kwargs)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
