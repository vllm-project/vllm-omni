# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Camera trajectory loading and geometry for LingBot World conditioning."""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

_REFERENCE_HEIGHT = 480
_REFERENCE_WIDTH = 832
_MAX_ACTION_FRAMES = 117
_MAX_NPY_HEADER_BYTES = 16 * 1024
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW


@dataclass(frozen=True)
class CameraTrajectory:
    """Camera-to-world poses and ``(fx, fy, cx, cy)`` intrinsics by frame."""

    poses: torch.Tensor
    intrinsics: torch.Tensor


@dataclass(frozen=True)
class TrustedActionDirectory:
    """Canonical action location tied to the trusted root's filesystem identity."""

    root: Path
    relative: Path
    root_device: int
    root_inode: int


def resolve_trusted_action_directory(
    action_path: str | os.PathLike[str],
    trusted_root: str | os.PathLike[str],
) -> TrustedActionDirectory:
    """Resolve an action directory beneath a trusted root without exposing paths."""

    try:
        root = Path(trusted_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The configured LingBot trusted action root is unavailable.") from None
    if not root.is_dir():
        raise ValueError("The configured LingBot trusted action root must be a directory.")

    try:
        candidate = Path(action_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve(strict=True)
        relative = candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise ValueError(
            "sampling_params.extra_args.action_path must be contained by the trusted action root."
        ) from None
    if not candidate.is_dir():
        raise ValueError("sampling_params.extra_args.action_path must identify a directory in the trusted action root.")

    try:
        root_stat = root.stat()
    except OSError:
        raise ValueError("The configured LingBot trusted action root is unavailable.") from None
    return TrustedActionDirectory(
        root=root,
        relative=relative,
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
    )


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


@contextmanager
def _open_action_directory(action_directory: TrustedActionDirectory) -> Iterator[int]:
    root_fd: int | None = None
    action_fd: int | None = None
    try:
        root_fd = os.open(action_directory.root, _DIRECTORY_OPEN_FLAGS)
        root_stat = os.fstat(root_fd)
        if (root_stat.st_dev, root_stat.st_ino) != (
            action_directory.root_device,
            action_directory.root_inode,
        ):
            raise ValueError("The configured LingBot trusted action root changed before camera files were opened.")

        action_fd = os.dup(root_fd)
        for component in action_directory.relative.parts:
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=action_fd)
            os.close(action_fd)
            action_fd = next_fd
    except OSError:
        if action_fd is not None:
            os.close(action_fd)
        if root_fd is not None:
            os.close(root_fd)
        raise ValueError("Unable to securely open the LingBot action directory within the trusted root.") from None
    except ValueError:
        if action_fd is not None:
            os.close(action_fd)
        if root_fd is not None:
            os.close(root_fd)
        raise
    assert action_fd is not None
    assert root_fd is not None
    try:
        yield action_fd
    finally:
        os.close(action_fd)
        os.close(root_fd)


@contextmanager
def _open_camera_file(action_fd: int, filename: str) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    file_fd = -1
    try:
        file_fd = os.open(filename, _FILE_OPEN_FLAGS, dir_fd=action_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{filename} must be a regular NPY file.")
        with os.fdopen(file_fd, "rb", closefd=True) as file_handle:
            file_fd = -1
            yield file_handle, file_stat
    except FileNotFoundError:
        raise FileNotFoundError(f"camera trajectory file not found: {filename}") from None
    except OSError:
        raise ValueError(f"{filename} could not be opened securely from the trusted action directory.") from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def _load_bounded_npy(
    action_fd: int,
    filename: str,
    *,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    with _open_camera_file(action_fd, filename) as (file_handle, file_stat):
        try:
            version = np.lib.format.read_magic(file_handle)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    file_handle,
                    max_header_size=_MAX_NPY_HEADER_BYTES,
                )
            elif version == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    file_handle,
                    max_header_size=_MAX_NPY_HEADER_BYTES,
                )
            else:
                raise ValueError(f"{filename} must use NPY format version 1.0 or 2.0.")
        except (EOFError, TypeError, ValueError):
            raise ValueError(f"{filename} has an invalid or oversized NPY header.") from None

        if len(shape) != len(trailing_shape) + 1 or tuple(shape[1:]) != trailing_shape:
            expected = ", ".join(str(value) for value in trailing_shape)
            raise ValueError(f"{filename} must have shape [frames, {expected}], got {shape}.")
        frames = shape[0]
        if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= _MAX_ACTION_FRAMES:
            raise ValueError(f"{filename} must contain between 1 and {_MAX_ACTION_FRAMES} frames.")
        if dtype.kind not in "fiu" or dtype.hasobject or dtype.itemsize > 8:
            raise ValueError(f"{filename} must contain real numeric values with at most 8 bytes per element.")

        data_offset = file_handle.tell()
        expected_bytes = math.prod(shape) * dtype.itemsize
        if file_stat.st_size != data_offset + expected_bytes:
            raise ValueError(f"{filename} byte size does not match its NPY header.")
        file_handle.seek(0)
        try:
            array = np.load(file_handle, allow_pickle=False)
        except (EOFError, TypeError, ValueError):
            raise ValueError(f"{filename} does not contain a valid numeric NPY array.") from None
    result: np.ndarray = np.asarray(array, dtype=np.float32)
    return result


def load_camera_trajectory(action_directory: TrustedActionDirectory) -> CameraTrajectory:
    """Load bounded numeric camera arrays through a trusted directory handle."""

    if not isinstance(action_directory, TrustedActionDirectory):
        raise TypeError("action_directory must be resolved against a trusted action root before loading.")
    with _open_action_directory(action_directory) as action_fd:
        poses_array = _load_bounded_npy(action_fd, "poses.npy", trailing_shape=(4, 4))
        intrinsics_array = _load_bounded_npy(action_fd, "intrinsics.npy", trailing_shape=(4,))

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


def _rotation_matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to normalized ``(w, x, y, z)`` quaternions."""

    quaternions = np.empty((matrices.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
                quaternion = np.array(
                    [
                        (matrix[2, 1] - matrix[1, 2]) / scale,
                        0.25 * scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
                quaternion = np.array(
                    [
                        (matrix[0, 2] - matrix[2, 0]) / scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        0.25 * scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                    ]
                )
            else:
                scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
                quaternion = np.array(
                    [
                        (matrix[1, 0] - matrix[0, 1]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            raise ValueError("camera rotation matrices must describe valid rotations")
        quaternions[index] = quaternion / norm
    return quaternions


def _quaternions_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    matrices: np.ndarray = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    return matrices


def _slerp_rotations(matrices: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    quaternions = _rotation_matrices_to_quaternions(matrices)
    interpolated = np.empty((target_times.shape[0], 4), dtype=np.float64)
    for target_index, target_time in enumerate(target_times):
        scaled_time = float(target_time) * (matrices.shape[0] - 1)
        left = min(max(math.floor(scaled_time), 0), matrices.shape[0] - 2)
        fraction = scaled_time - left
        first = quaternions[left]
        second = quaternions[left + 1]
        cosine = float(np.dot(first, second))
        if cosine < 0.0:
            second = -second
            cosine = -cosine
        cosine = min(cosine, 1.0)
        if cosine > 0.9995:
            quaternion = first + fraction * (second - first)
            interpolated[target_index] = quaternion / np.linalg.norm(quaternion)
            continue
        angle = math.acos(cosine)
        denominator = math.sin(angle)
        interpolated[target_index] = (
            math.sin((1.0 - fraction) * angle) / denominator * first + math.sin(fraction * angle) / denominator * second
        )
    return _quaternions_to_rotation_matrices(interpolated)


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

    # Rotations use quaternion SLERP; translations and intrinsics are linear.
    target_times = np.linspace(0.0, 1.0, num_frames)
    rotation_matrices = trajectory.poses[:, :3, :3].detach().cpu().double().numpy()
    interpolated_rotations = _slerp_rotations(rotation_matrices, target_times)
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
    """Invert rigid transforms using R^-1 = R^T and t^-1 = -R^T t."""

    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3:]
    inverse_rotations = rotations.transpose(-1, -2)
    inverse = torch.eye(4, device=poses.device, dtype=poses.dtype).repeat(poses.shape[0], 1, 1)
    inverse[:, :3, :3] = inverse_rotations
    inverse[:, :3, 3:] = -torch.bmm(inverse_rotations, translations)
    return inverse


def _prepare_framewise_poses(raw_poses: torch.Tensor) -> torch.Tensor:
    """Convert raw C2W poses to normalized first-relative framewise deltas."""

    # Remove global placement, then convert the sequence to framewise deltas.
    first_world_to_camera = _invert_camera_poses(raw_poses[:1])
    relative_poses = torch.matmul(first_world_to_camera, raw_poses).clone()
    relative_poses[0] = torch.eye(4, device=raw_poses.device, dtype=raw_poses.dtype)
    if relative_poses.shape[0] > 1:
        relative_poses[1:] = torch.bmm(
            _invert_camera_poses(relative_poses[:-1]),
            relative_poses[1:],
        )

    # Only relative direction/magnitude matters to the checkpoint; normalizing
    # by the largest step removes the trajectory's arbitrary translation scale.
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

    # Compute geometry in at least FP32, then cast at the model boundary.
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

    # Pinhole back-projection: pixel (x, y) becomes the camera-space direction
    # ((x-cx)/fx, (y-cy)/fy, 1), which is normalized before rotation.
    fx, fy, cx, cy = intrinsics.unbind(dim=1)
    camera_x = (grid_x - cx[:, None, None]) / fx[:, None, None]
    camera_y = (grid_y - cy[:, None, None]) / fy[:, None, None]
    camera_directions = torch.stack(
        (camera_x, camera_y, torch.ones_like(camera_x)),
        dim=-1,
    )
    camera_directions = camera_directions / torch.linalg.vector_norm(camera_directions, dim=-1, keepdim=True)

    # A ray is represented by its camera center (origin) and unit direction in
    # the prepared relative coordinate frame. Channel order is checkpoint
    # data, not an implementation preference: [origin_xyz, direction_xyz].
    world_directions = torch.einsum(
        "fij,fhwj->fhwi",
        poses[:, :3, :3],
        camera_directions,
    )
    world_directions = world_directions / torch.linalg.vector_norm(world_directions, dim=-1, keepdim=True)
    ray_origins = poses[:, None, None, :3, 3].expand_as(world_directions)

    embedding = torch.cat((ray_origins, world_directions), dim=-1)
    return embedding.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)
