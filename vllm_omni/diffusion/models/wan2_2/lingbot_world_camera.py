# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure camera-geometry helpers for LingBot World video conditioning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

_REFERENCE_HEIGHT = 480
_REFERENCE_WIDTH = 832


@dataclass(frozen=True)
class CameraTrajectory:
    """Camera poses and pinhole intrinsics ordered by video frame."""

    poses: torch.Tensor
    intrinsics: torch.Tensor


def _validate_trajectory(
    trajectory: CameraTrajectory,
    *,
    file_suffix: str = "",
) -> None:
    poses = trajectory.poses
    intrinsics = trajectory.intrinsics
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses{file_suffix} must have shape [frames, 4, 4], got {tuple(poses.shape)}")
    if intrinsics.ndim != 2 or intrinsics.shape[1:] != (4,):
        raise ValueError(
            f"intrinsics{file_suffix} must have shape [frames, 4] ordered as fx, fy, cx, cy, "
            f"got {tuple(intrinsics.shape)}"
        )
    if poses.shape[0] == 0:
        raise ValueError("camera trajectory must contain at least one frame")
    if poses.shape[0] != intrinsics.shape[0]:
        raise ValueError("poses and intrinsics must contain the same number of frames")
    if not torch.isfinite(poses).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("camera trajectory values must all be finite")


def load_camera_trajectory(action_path: str | os.PathLike[str]) -> CameraTrajectory:
    """Load raw camera-to-world poses and intrinsics from an action directory."""

    action_directory = Path(action_path)
    poses_path = action_directory / "poses.npy"
    intrinsics_path = action_directory / "intrinsics.npy"
    for required_path in (poses_path, intrinsics_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"camera trajectory file not found: {required_path}")

    try:
        poses_array = np.asarray(np.load(poses_path, allow_pickle=False), dtype=np.float32)
        intrinsics_array = np.asarray(np.load(intrinsics_path, allow_pickle=False), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera trajectory arrays must contain numeric values") from exc

    trajectory = CameraTrajectory(
        poses=torch.from_numpy(poses_array),
        intrinsics=torch.from_numpy(intrinsics_array),
    )
    _validate_trajectory(trajectory, file_suffix=".npy")
    return trajectory


def _linear_interpolate(values: torch.Tensor, target_times: np.ndarray) -> torch.Tensor:
    source_times = np.linspace(0.0, 1.0, values.shape[0])
    values_array = values.detach().cpu().double().numpy()
    columns = [
        np.interp(target_times, source_times, values_array[:, column]) for column in range(values_array.shape[1])
    ]
    interpolated = torch.from_numpy(np.stack(columns, axis=1))
    return interpolated.to(device=values.device, dtype=values.dtype)


def interpolate_camera_trajectory(
    trajectory: CameraTrajectory,
    num_frames: int,
) -> CameraTrajectory:
    """Resample translations/intrinsics linearly and rotations with quaternion SLERP."""

    _validate_trajectory(trajectory)
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    source_frames = trajectory.poses.shape[0]
    if num_frames == source_frames:
        return trajectory
    if source_frames == 1:
        return CameraTrajectory(
            poses=trajectory.poses.repeat(num_frames, 1, 1),
            intrinsics=trajectory.intrinsics.repeat(num_frames, 1),
        )

    source_times = np.linspace(0.0, 1.0, source_frames)
    target_times = np.linspace(0.0, 1.0, num_frames)
    rotations = Rotation.from_matrix(trajectory.poses[:, :3, :3].detach().cpu().double().numpy())
    interpolated_rotations = Slerp(source_times, rotations)(target_times).as_matrix()
    translations = _linear_interpolate(trajectory.poses[:, :3, 3], target_times)
    intrinsics = _linear_interpolate(trajectory.intrinsics, target_times)

    poses = torch.eye(
        4,
        device=trajectory.poses.device,
        dtype=trajectory.poses.dtype,
    ).repeat(num_frames, 1, 1)
    poses[:, :3, :3] = torch.from_numpy(interpolated_rotations).to(
        device=trajectory.poses.device,
        dtype=trajectory.poses.dtype,
    )
    poses[:, :3, 3] = translations
    return CameraTrajectory(poses=poses, intrinsics=intrinsics)


def _invert_camera_poses(poses: torch.Tensor) -> torch.Tensor:
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3:]
    inverse_rotations = rotations.transpose(-1, -2)
    inverse = torch.eye(4, device=poses.device, dtype=poses.dtype).repeat(poses.shape[0], 1, 1)
    inverse[:, :3, :3] = inverse_rotations
    inverse[:, :3, 3:] = -torch.bmm(inverse_rotations, translations)
    return inverse


def _prepare_framewise_poses(raw_poses: torch.Tensor) -> torch.Tensor:
    """Convert raw C2W poses to normalized first-relative framewise deltas."""

    first_world_to_camera = _invert_camera_poses(raw_poses[:1])
    relative_poses = torch.matmul(first_world_to_camera, raw_poses).clone()
    relative_poses[0] = torch.eye(4, device=raw_poses.device, dtype=raw_poses.dtype)
    if relative_poses.shape[0] > 1:
        relative_poses[1:] = torch.bmm(
            _invert_camera_poses(relative_poses[:-1]),
            relative_poses[1:],
        )

    translations = relative_poses[:, :3, 3]
    max_translation_norm = torch.linalg.vector_norm(translations, dim=-1).max()
    if max_translation_norm > 0:
        relative_poses[:, :3, 3] = translations / max_translation_norm
    return relative_poses


def build_plucker_embedding(
    trajectory: CameraTrajectory,
    *,
    height: int,
    width: int,
    target_height: int,
    target_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build checkpoint-ordered ``(ray origin, ray direction)`` channels."""

    _validate_trajectory(trajectory)
    dimensions = {
        "height": height,
        "width": width,
        "target_height": target_height,
        "target_width": target_width,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"dtype must be floating point, got {dtype}")

    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    poses = _prepare_framewise_poses(trajectory.poses.to(device=device, dtype=compute_dtype))
    intrinsics = trajectory.intrinsics.to(device=device, dtype=compute_dtype).clone()
    intrinsics[:, (0, 2)] *= width / _REFERENCE_WIDTH
    intrinsics[:, (1, 3)] *= height / _REFERENCE_HEIGHT
    if torch.any(intrinsics[:, :2] == 0):
        raise ValueError("camera focal lengths fx and fy must be non-zero")

    # Sample requested-pixel coordinates directly at the conditioning resolution.
    x_coordinates = (torch.arange(target_width, device=device, dtype=compute_dtype) + 0.5) * (width / target_width)
    y_coordinates = (torch.arange(target_height, device=device, dtype=compute_dtype) + 0.5) * (height / target_height)
    grid_y, grid_x = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    fx, fy, cx, cy = intrinsics.unbind(dim=1)
    camera_x = (grid_x - cx[:, None, None]) / fx[:, None, None]
    camera_y = (grid_y - cy[:, None, None]) / fy[:, None, None]
    camera_directions = torch.stack(
        (camera_x, camera_y, torch.ones_like(camera_x)),
        dim=-1,
    )
    camera_directions = camera_directions / torch.linalg.vector_norm(camera_directions, dim=-1, keepdim=True)

    world_directions = torch.einsum(
        "fij,fhwj->fhwi",
        poses[:, :3, :3],
        camera_directions,
    )
    world_directions = world_directions / torch.linalg.vector_norm(world_directions, dim=-1, keepdim=True)
    ray_origins = poses[:, None, None, :3, 3].expand_as(world_directions)

    embedding = torch.cat((ray_origins, world_directions), dim=-1)
    return embedding.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)
