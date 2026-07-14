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
resolve_trusted_action_directory = _CAMERA_MODULE.resolve_trusted_action_directory

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


def _trusted(directory: Path):
    return resolve_trusted_action_directory(".", directory)


def _trajectory(
    poses: torch.Tensor,
    intrinsics: torch.Tensor | None = None,
) -> CameraTrajectory:
    if intrinsics is None:
        intrinsics = torch.tensor([[832.0 / 3.0, 160.0, 416.0, 240.0]]).repeat(poses.shape[0], 1)
    return CameraTrajectory(poses=poses, intrinsics=intrinsics)


@pytest.mark.parametrize("present_file", [None, "poses.npy"])
def test_load_camera_trajectory_rejects_missing_files(tmp_path: Path, present_file: str | None) -> None:
    if present_file == "poses.npy":
        np.save(tmp_path / present_file, _identity_poses(1))

    missing_file = "poses.npy" if present_file is None else "intrinsics.npy"
    with pytest.raises(FileNotFoundError, match=missing_file):
        load_camera_trajectory(_trusted(tmp_path))


@pytest.mark.parametrize(
    ("poses", "intrinsics", "message"),
    [
        (np.zeros((2, 3, 4)), np.zeros((2, 4)), "poses.npy"),
        (_identity_poses(2), np.zeros((2, 3)), "intrinsics.npy"),
        (_identity_poses(0), np.zeros((0, 4)), "1 and 117"),
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
        load_camera_trajectory(_trusted(tmp_path))


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
        load_camera_trajectory(_trusted(tmp_path))


def test_load_camera_trajectory_preserves_raw_c2w_poses(tmp_path: Path) -> None:
    first_c2w = np.array(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    second_c2w = np.array(
        [
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 1.0, 0.0, 4.0],
            [-1.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    poses = np.stack((first_c2w, second_c2w))
    intrinsics = np.array([[100.0, 110.0, 120.0, 130.0]] * 2)
    _write_trajectory(tmp_path, poses, intrinsics)

    trajectory = load_camera_trajectory(_trusted(tmp_path))

    assert trajectory.poses.dtype == torch.float32
    assert trajectory.intrinsics.dtype == torch.float32
    torch.testing.assert_close(trajectory.poses, torch.from_numpy(poses).float())
    torch.testing.assert_close(trajectory.intrinsics, torch.from_numpy(intrinsics).float())


def test_load_camera_trajectory_rejects_object_dtype_before_materializing(tmp_path: Path) -> None:
    poses = _identity_poses(1).astype(object)
    intrinsics = np.ones((1, 4), dtype=np.float32)
    _write_trajectory(tmp_path, poses, intrinsics)

    with pytest.raises(ValueError, match="poses.npy.*real numeric"):
        load_camera_trajectory(_trusted(tmp_path))


def test_load_camera_trajectory_rejects_frame_count_above_request_limit(tmp_path: Path) -> None:
    _write_trajectory(
        tmp_path,
        _identity_poses(118),
        np.ones((118, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="poses.npy.*1 and 117 frames"):
        load_camera_trajectory(_trusted(tmp_path))


def test_load_camera_trajectory_rejects_truncated_payload_before_loading(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, _identity_poses(1), np.ones((1, 4), dtype=np.float32))
    poses_path = tmp_path / "poses.npy"
    poses_path.write_bytes(poses_path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="poses.npy byte size"):
        load_camera_trajectory(_trusted(tmp_path))


def test_load_camera_trajectory_rejects_oversized_npy_header(tmp_path: Path) -> None:
    header_size = 20_000
    (tmp_path / "poses.npy").write_bytes(
        np.lib.format.magic(2, 0) + header_size.to_bytes(4, "little") + b" " * (header_size - 1) + b"\n"
    )
    np.save(tmp_path / "intrinsics.npy", np.ones((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="poses.npy.*oversized NPY header"):
        load_camera_trajectory(_trusted(tmp_path))


def test_load_camera_trajectory_rejects_symlink_swap_after_resolution(tmp_path: Path) -> None:
    trusted_root = tmp_path / "trusted"
    action_directory = trusted_root / "action"
    outside = tmp_path / "outside"
    action_directory.mkdir(parents=True)
    outside.mkdir()
    _write_trajectory(action_directory, _identity_poses(1), np.ones((1, 4), dtype=np.float32))
    _write_trajectory(outside, _identity_poses(1), np.ones((1, 4), dtype=np.float32))
    resolved = resolve_trusted_action_directory("action", trusted_root)
    action_directory.rename(trusted_root / "original-action")
    action_directory.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="securely open.*action directory"):
        load_camera_trajectory(resolved)


def test_load_camera_trajectory_rejects_symlinked_camera_file(tmp_path: Path) -> None:
    trusted_root = tmp_path / "trusted"
    action_directory = trusted_root / "action"
    outside = tmp_path / "outside"
    action_directory.mkdir(parents=True)
    outside.mkdir()
    _write_trajectory(action_directory, _identity_poses(1), np.ones((1, 4), dtype=np.float32))
    np.save(outside / "poses.npy", _identity_poses(1))
    resolved = resolve_trusted_action_directory("action", trusted_root)
    (action_directory / "poses.npy").unlink()
    (action_directory / "poses.npy").symlink_to(outside / "poses.npy")

    with pytest.raises(ValueError, match="poses.npy.*opened securely"):
        load_camera_trajectory(resolved)


def test_load_camera_trajectory_ignores_non_camera_event_files(tmp_path: Path) -> None:
    poses = _identity_poses(1)
    intrinsics = np.ones((1, 4), dtype=np.float32)
    _write_trajectory(tmp_path, poses, intrinsics)
    np.save(tmp_path / "wasd_events.npy", np.array([{"key": "w"}], dtype=object))

    trajectory = load_camera_trajectory(_trusted(tmp_path))

    torch.testing.assert_close(trajectory.poses, torch.from_numpy(poses).float())


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

    torch.testing.assert_close(result[0, :3, 1, 1], torch.zeros(3))
    torch.testing.assert_close(result[0, 3:, 1, 1], torch.tensor([0.0, 0.0, 1.0]))


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

    expected_corner_direction = torch.tensor([-0.875, -0.75, 1.0])
    expected_corner_direction /= torch.linalg.vector_norm(expected_corner_direction)
    torch.testing.assert_close(result[0, 3:, 0, 0], expected_corner_direction)


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

    # Target centers map through the source scale: (0.5 * 2, 0.5 * 3).
    expected_direction = torch.tensor([-0.8, -0.5, 1.0])
    expected_direction /= torch.linalg.vector_norm(expected_direction)
    torch.testing.assert_close(result[0, 3:, 0, 0], expected_direction)


def test_build_plucker_embedding_uses_framewise_deltas_for_rotations_and_translations() -> None:
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[0, :3, 3] = torch.tensor([5.0, 7.0, 11.0])
    poses[1, :3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    poses[1, :3, 3] = torch.tensor([5.0, 9.0, 11.0])
    poses[2, :3, :3] = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    poses[2, :3, 3] = torch.tensor([3.0, 9.0, 11.0])
    trajectory = _trajectory(
        poses,
        torch.tensor([[416.0, 480.0, 208.0, 240.0]]).repeat(3, 1),
    )

    result = build_plucker_embedding(
        trajectory,
        height=1,
        width=2,
        target_height=1,
        target_width=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_direction = torch.tensor([0.0, 1.0, 1.0]) / math.sqrt(2.0)
    torch.testing.assert_close(result[0, :3, 0, 1], torch.zeros(3))
    torch.testing.assert_close(result[1, :3, 0, 1], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(result[2, :3, 0, 1], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(result[1, 3:, 0, 1], expected_direction)
    torch.testing.assert_close(result[2, 3:, 0, 1], expected_direction)


def test_build_plucker_embedding_normalizes_framewise_translation_by_max_norm() -> None:
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, :3, 3] = torch.tensor(
        [
            [10.0, -2.0, 5.0],
            [13.0, 2.0, 5.0],
            [19.0, 10.0, 5.0],
        ]
    )

    result = build_plucker_embedding(
        _trajectory(poses),
        height=1,
        width=1,
        target_height=1,
        target_width=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_origins = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.6, 0.8, 0.0]])
    torch.testing.assert_close(result[:, :3, 0, 0], expected_origins)


def test_build_plucker_embedding_interpolates_raw_poses_before_framewise_conversion() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[0, :3, 3] = torch.tensor([20.0, 4.0, -2.0])
    poses[1, :3, 3] = torch.tensor([26.0, 12.0, -2.0])
    raw = _trajectory(poses)

    interpolated = interpolate_camera_trajectory(raw, 3)
    result = build_plucker_embedding(
        interpolated,
        height=1,
        width=1,
        target_height=1,
        target_width=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_delta = torch.tensor([0.6, 0.8, 0.0])
    torch.testing.assert_close(result[0, :3, 0, 0], torch.zeros(3))
    torch.testing.assert_close(result[1, :3, 0, 0], expected_delta)
    torch.testing.assert_close(result[2, :3, 0, 0], expected_delta)


def test_build_plucker_embedding_uses_half_pixel_centers_and_origin_direction_channel_order() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, 3] = torch.tensor([2.0, 0.0, 0.0])
    trajectory = _trajectory(
        poses,
        torch.tensor([[832.0, 480.0, 0.0, 0.0]]).repeat(2, 1),
    )

    result = build_plucker_embedding(
        trajectory,
        height=1,
        width=2,
        target_height=1,
        target_width=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_direction = torch.tensor([0.25, 0.5, 1.0])
    expected_direction /= torch.linalg.vector_norm(expected_direction)
    torch.testing.assert_close(result[1, :3, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(result[1, 3:, 0, 0], expected_direction)


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
