# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas (Copyright 2026 Tencent).

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def camera_boundaries(bounds, frames):
    count = (frames - 1) // 4 * 4 + 1
    result = [min(max(0, int(value) // 4 * 4 - 3), count) for value in bounds]
    result[0], result[-1] = 0, count
    for index in range(1, len(result)):
        if result[index] <= result[index - 1]:
            result[index] = min(result[index - 1] + 1, count)
    return result if all(a < b for a, b in zip(result, result[1:])) else [0, count]


def normalize_poses(
    poses,
    bounds,
    *,
    normalize_relative_to_first_frame=True,
    normalize_framewise=True,
    normalize_displacement=True,
    normalize_displacement_per_video=False,
):
    if normalize_displacement_per_video and not normalize_displacement:
        raise ValueError("Camera per-video displacement requires displacement normalization.")
    if normalize_displacement and not normalize_framewise:
        raise ValueError("Camera displacement requires framewise normalization.")
    if normalize_framewise and not normalize_relative_to_first_frame:
        raise ValueError("Camera framewise normalization requires first-frame normalization.")
    if not normalize_relative_to_first_frame:
        return poses.copy()
    output = np.zeros_like(poses)
    identity = np.eye(4, dtype=poses.dtype)

    def scale(values):
        raw = float(np.quantile(np.linalg.norm(values, axis=1), 0.99)) if len(values) else 0.0
        if not np.isfinite(raw) or raw < 0:
            raise ValueError("Invalid camera displacement scale.")
        return max(raw, 1e-6)

    for start, end in zip(bounds, bounds[1:]):
        relative = identity @ np.linalg.inv(poses[start])
        output[start] = identity
        if end - start > 1:
            output[start + 1 : end] = relative @ poses[start + 1 : end]
        if normalize_framewise and end - start > 1:
            integral = output[start:end].copy()
            inverse = np.zeros_like(integral[:-1])
            inverse[:, :3, :3] = np.swapaxes(integral[:-1, :3, :3], -1, -2)
            inverse[:, :3, 3:4] = -np.matmul(inverse[:, :3, :3], integral[:-1, :3, 3:4])
            inverse[:, 3, 3] = 1.0
            output[start + 1 : end] = np.matmul(inverse, integral[1:])
            if normalize_displacement and not normalize_displacement_per_video:
                output[start:end, :3, 3] /= scale(output[start + 1 : end, :3, 3])
    if normalize_displacement and normalize_displacement_per_video and len(output) > 1:
        parts = [output[start + 1 : end, :3, 3] for start, end in zip(bounds, bounds[1:]) if end - start > 1]
        output[:, :3, 3] /= scale(np.concatenate(parts) if parts else output[:0, :3, 3])
    return output


def _anchor(poses):
    relative = np.eye(4, dtype=np.int64) @ np.linalg.inv(poses[0])
    result = np.zeros_like(poses)
    result[0] = np.eye(4)
    result[1:] = relative @ poses[1:]
    return result


def _resample(poses, count):
    if count < 1 or not len(poses):
        raise ValueError("Camera timeline must be nonempty.")
    if len(poses) == count:
        return poses.copy()
    if len(poses) == 1:
        return np.repeat(poses, count, axis=0)
    source, target = np.linspace(0, 1, len(poses)), np.linspace(0, 1, count)
    result = np.repeat(np.eye(4)[None], count, axis=0)
    result[:, :3, :3] = Slerp(source, Rotation.from_matrix(poses[:, :3, :3]))(target).as_matrix()
    result[:, :3, 3] = np.stack([np.interp(target, source, poses[:, index, 3]) for index in range(3)], axis=1)
    return result.astype(poses.dtype)


def _inline_poses(payload):
    frames = payload["frames"] if isinstance(payload, dict) else payload
    result = []
    for frame in frames:
        position = np.array([frame["position"][axis] for axis in "xyz"], dtype=np.float64)
        quaternion = np.array([frame["quaternion"][axis] for axis in "xyzw"], dtype=np.float64)
        length = np.linalg.norm(quaternion)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or length < 1e-8:
            raise ValueError("Invalid camera position/quaternion.")
        fov = float(frame.get("fov", 55.0))
        if not 0 < fov < 180:
            raise ValueError("Camera FOV must be between zero and 180 degrees.")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(quaternion / length).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = position
        result.append(pose)
    if not result:
        raise ValueError("Camera recording is empty.")
    return np.stack(result)


def _validate_intrinsics(values, frames):
    values = np.asarray(values)
    valid_shape = (
        values.ndim == 1
        and values.shape == (4,)
        or values.ndim == 2
        and values.shape[1] == 4
        and values.shape[0] in (1, frames)
        or values.ndim == 3
        and values.shape[1:] == (3, 3)
        and values.shape[0] in (1, frames)
    )
    if not valid_shape or not np.isfinite(values).all():
        raise ValueError("Camera intrinsics must be finite [4], [1/T,4] or [1/T,3,3] arrays.")


def _precomputed(data, frames, bounds, flags):
    if str(data["camera_format"].item()) != "demo_camera_precomputed_v1":
        raise ValueError("Unsupported precomputed camera format.")
    if json.loads(str(data["normalization_json"].item())) != flags:
        raise ValueError("Precomputed camera normalization mismatch.")
    stored = data["segment_boundaries_pixel"].tolist()
    offsets, lengths = data["turn_camera_offsets"].tolist(), data["turn_num_frames"].tolist()
    enabled = data["turn_has_camera"].tolist()
    poses = data["norm_extrinsics"].copy()
    if (
        stored != list(bounds or [])
        or not stored
        or stored[-1] != frames
        or len(stored) != len(lengths) + 1
        or len(offsets) != len(stored)
        or len(enabled) != len(lengths)
        or offsets[0] != 0
        or offsets[-1] != len(poses)
        or stored[0] != 0
    ):
        raise ValueError("Precomputed camera requires its original turn boundaries and length.")
    for index, length in enumerate(lengths):
        latent_count = (int(length) - 1) // 4 + 1
        pixels = 4 * latent_count if index < len(lengths) - 1 else length
        if length <= 0 or offsets[index + 1] - offsets[index] != (length - 1) // 4 * 4 + 1:
            raise ValueError("Invalid precomputed camera turn length.")
        if stored[index + 1] - stored[index] != pixels:
            raise ValueError("Precomputed camera latent mapping does not match the output timeline.")
    for name, shape in (
        ("norm_extrinsics", (len(poses), 4, 4)),
        ("norm_intrinsics", (len(poses), 3, 3)),
        ("action_labels", (len(poses),)),
    ):
        value = data[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid precomputed {name}.")
        if name == "action_labels" and (not np.issubdtype(value.dtype, np.integer) or np.any(value < 0)):
            raise ValueError("Invalid precomputed action labels.")
    return poses, offsets


def prepare_camera_inputs(
    trajectory_source,
    num_frames,
    device,
    dtype,
    *,
    normalize_relative_to_first_frame=False,
    normalize_framewise=False,
    normalize_displacement=False,
    normalize_displacement_per_video=False,
    segment_boundaries_pixel=None,
    translation_scale=1.0,
    rotation_scale=1.0,
):
    flags = dict(
        normalize_relative_to_first_frame=normalize_relative_to_first_frame,
        normalize_framewise=normalize_framewise,
        normalize_displacement=normalize_displacement,
        normalize_displacement_per_video=normalize_displacement_per_video,
    )
    if (
        not np.isfinite(translation_scale)
        or translation_scale <= 0
        or not np.isfinite(rotation_scale)
        or rotation_scale <= 0
    ):
        raise ValueError("Camera gains must be finite positive values.")
    count = (num_frames - 1) // 4 * 4 + 1
    path = Path(trajectory_source)
    offsets = None
    precomputed = False
    demo_bounds = None
    if path.suffix == ".npz" and path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if "camera_format" in data:
                poses, offsets = _precomputed(data, num_frames, segment_boundaries_pixel, flags)
                precomputed = True
            else:
                poses = data["c2ws"].copy()
                _validate_intrinsics(data["intrinsics"], len(poses))
                num_frames = len(poses)
                count = (num_frames - 1) // 4 * 4 + 1
                demo_bounds = data["segment_boundaries_pixel"].tolist() if "segment_boundaries_pixel" in data else None
    else:
        inline = trajectory_source.lstrip().startswith(("{", "["))
        is_preset = False
        if inline:
            payload = json.loads(trajectory_source)
        elif path.is_file():
            payload = json.loads(path.read_text())
        else:
            is_preset = True
            presets = json.loads(Path(__file__).with_name("camera_trajectories.json").read_text())
            if trajectory_source not in presets:
                raise FileNotFoundError(f"Unknown Hy-Nomas camera trajectory: {trajectory_source}")
            payload = {
                str(index): {"extrinsic": np.asarray(matrix).reshape(4, 4)}
                for index, matrix in enumerate(presets[trajectory_source])
            }
        if isinstance(payload, list) or isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            poses = _inline_poses(payload)
        else:
            if not is_preset:
                rows = [payload[key] for key in sorted(payload, key=lambda key: int(key))]
                intrinsic_key = "intrinsic" if "intrinsic" in rows[0] else "K"
                _validate_intrinsics(np.stack([row[intrinsic_key] for row in rows]), len(rows))
            poses = np.stack(
                [np.asarray(payload[key]["extrinsic"]) for key in sorted(payload, key=lambda key: int(key))]
            ).copy()
        poses = _anchor(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Camera poses must be finite nonempty [T,4,4] matrices.")
    if not precomputed:
        bounds = camera_boundaries(segment_boundaries_pixel or [0, num_frames], num_frames)
        if demo_bounds is None:
            poses = _resample(poses, count)
        else:
            if (
                demo_bounds[0] != 0
                or demo_bounds[-1] != len(poses)
                or any(a >= b for a, b in zip(demo_bounds, demo_bounds[1:]))
            ):
                raise ValueError("Invalid demo NPZ segment boundaries.")
            bounds = camera_boundaries(demo_bounds, num_frames)
            if len(bounds) != len(demo_bounds):
                raise ValueError("Camera segments are too short for the latent timeline.")
            poses = np.concatenate(
                [
                    _resample(poses[left:right], end - start)
                    for left, right, start, end in zip(demo_bounds, demo_bounds[1:], bounds, bounds[1:])
                ]
            )
        if normalize_relative_to_first_frame:
            poses = normalize_poses(poses.astype(np.float64), bounds, **flags).astype(np.float32)
    if not np.isfinite(poses).all() or np.abs(poses).max() > 1e6:
        raise ValueError("Camera normalization produced invalid pose magnitudes.")
    if translation_scale != 1.0:
        poses = poses.copy()
        poses[:, :3, 3] *= translation_scale
    if rotation_scale != 1.0:
        if not normalize_relative_to_first_frame or not normalize_framewise:
            raise ValueError("Camera rotation gains require framewise-relative poses.")
        rotations = poses[:, :3, :3].astype(np.float64)
        if not (
            np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-5, rtol=0)
            and np.allclose(np.linalg.det(rotations), 1.0, atol=2e-5, rtol=0)
        ):
            raise ValueError("Camera rotation gains require valid SO(3) matrices.")
        rotvecs = Rotation.from_matrix(rotations).as_rotvec() * rotation_scale
        if not np.isfinite(rotvecs).all() or np.linalg.norm(rotvecs, axis=1).max() >= np.pi - 1e-7:
            raise ValueError("Camera rotation gain reaches/exceeds 180 degrees.")
        poses = poses.copy()
        poses[:, :3, :3] = Rotation.from_rotvec(rotvecs).as_matrix()
    if not np.isfinite(poses).all():
        raise ValueError("Camera gains produced nonfinite poses.")
    result = {
        "norm_extrinsics": torch.from_numpy(poses).float()[None].to(device=device, dtype=dtype),
        "norm_intrinsics": None,
        "action_onehot": None,
        "num_frames": num_frames,
    }
    if offsets is not None:
        result["turn_camera_offsets"] = offsets
    return result
