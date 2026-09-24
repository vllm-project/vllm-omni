# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Camera-control helpers for SANA-WM.

This module mirrors the public NVlabs/Sana SANA-WM camera preparation path:
WASD/IJKL action rollout or explicit camera-to-world poses are converted into
relative poses, per-latent-frame ray metadata, and per-VAE-chunk Plucker maps.
Model-local projection layers consume these tensors in the Stage-1 DiT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

SANA_WM_DEFAULT_TRANSLATION_SPEED = 0.05
SANA_WM_DEFAULT_ROTATION_SPEED_DEG = 1.2
SANA_WM_DEFAULT_PITCH_LIMIT_DEG = 85.0
SANA_WM_ALLOWED_ACTION_KEYS = frozenset("wasdijkl")
SANA_WM_DEFAULT_VAE_STRIDE = (8, 32, 32)


@dataclass(frozen=True)
class SanaWmCameraCondition:
    poses: Any | None = None
    intrinsics: Any | None = None
    action: str | None = None
    num_frames: int | None = None
    height: int = 704
    width: int = 1280
    translation_speed: float = SANA_WM_DEFAULT_TRANSLATION_SPEED
    rotation_speed_deg: float = SANA_WM_DEFAULT_ROTATION_SPEED_DEG
    pitch_limit_deg: float = SANA_WM_DEFAULT_PITCH_LIMIT_DEG


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _parse_action_segments(action: str) -> list[tuple[list[str], int]]:
    """Parse an action string into ``(keys, duration)`` pairs without expanding it.

    A single segment names an unbounded rollout (``"w-1000000000"`` is valid
    syntax), so expanding here would allocate one list entry per frame before
    anything checks the frame count. Callers total the durations and validate
    against ``num_frames`` first; only :func:`_parse_action_string` expands.
    """
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("Sana-WM action string is empty.")

    segments: list[tuple[list[str], int]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(f"Invalid Sana-WM action segment {segment!r}: expected '<keys>-<duration>'.")
        keys_part, duration = segment.rsplit("-", 1)
        if not duration.isdigit() or int(duration) <= 0:
            raise ValueError(f"Sana-WM action segment {segment!r} has a non-positive duration {duration!r}.")
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list[str] = []
        else:
            bad = sorted({key for key in keys_lower if key not in SANA_WM_ALLOWED_ACTION_KEYS})
            if bad:
                allowed = "".join(sorted(SANA_WM_ALLOWED_ACTION_KEYS))
                raise ValueError(f"Sana-WM action segment {segment!r} contains unknown keys {bad}; allowed: {allowed}.")
            keys = sorted(set(keys_lower))
        segments.append((keys, int(duration)))
    return segments


def action_rollout_num_frames(action: str) -> int:
    """Frames ``action`` rolls out: the identity start pose plus one per step.

    Totals the durations arithmetically, so an oversized request is rejected
    before anything allocates per-frame state.
    """
    return sum(duration for _, duration in _parse_action_segments(action)) + 1


def _parse_action_string(action: str) -> list[list[str]]:
    per_frame: list[list[str]] = []
    for keys, duration in _parse_action_segments(action):
        per_frame.extend([keys] * duration)
    return per_frame


def action_string_to_c2w(
    action: str,
    *,
    translation_speed: float = SANA_WM_DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
) -> np.ndarray:
    """Roll out an OpenCV-convention ``(N + 1, 4, 4)`` c2w trajectory."""

    per_frame = _parse_action_string(action)
    rotate_rad = math.radians(rotation_speed_deg)
    pitch_limit_rad = math.radians(pitch_limit_deg)
    current = np.eye(4, dtype=np.float64)
    current_pitch = 0.0
    poses = [current.copy()]

    for keys in per_frame:
        held = set(keys)
        rotation = current[:3, :3]
        translation = current[:3, 3]

        pitch_delta = (rotate_rad if "i" in held else 0.0) - (rotate_rad if "k" in held else 0.0)
        new_pitch = current_pitch + pitch_delta
        if -pitch_limit_rad <= new_pitch <= pitch_limit_rad:
            current_pitch = new_pitch
        else:
            pitch_delta = 0.0

        yaw_delta = (rotate_rad if "l" in held else 0.0) - (rotate_rad if "j" in held else 0.0)
        new_rotation = _rot_y(yaw_delta) @ rotation @ _rot_x(pitch_delta)

        forward = new_rotation[:, 2].copy()
        forward[1] = 0.0
        right = new_rotation[:, 0].copy()
        right[1] = 0.0
        forward_norm = float(np.linalg.norm(forward))
        right_norm = float(np.linalg.norm(right))
        if forward_norm > 0:
            forward /= forward_norm + 1e-6
        if right_norm > 0:
            right /= right_norm + 1e-6

        move = np.zeros(3, dtype=np.float64)
        if "w" in held:
            move += forward * translation_speed
        if "s" in held:
            move -= forward * translation_speed
        if "d" in held:
            move += right * translation_speed
        if "a" in held:
            move -= right * translation_speed

        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = new_rotation
        current[:3, 3] = translation + move
        poses.append(current.copy())

    return np.stack(poses, axis=0).astype(np.float32)


def get_pose_inverse(transform: torch.Tensor) -> torch.Tensor:
    """Invert homogeneous rigid transforms with shape ``(..., 4, 4)``."""

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -torch.matmul(rotation_inv, translation.unsqueeze(-1)).squeeze(-1)
    output = torch.eye(4, dtype=transform.dtype, device=transform.device).repeat(transform.shape[:-2] + (1, 1))
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = translation_inv
    return output


def compute_raymap(
    intrinsics: torch.Tensor,
    poses: torch.Tensor,
    height: int,
    width: int,
    *,
    use_plucker: bool = True,
) -> torch.Tensor:
    """Compute SANA-WM geometry ray maps.

    Args:
        intrinsics: ``(T, 4)`` tensor in ``[fx, fy, cx, cy]`` order.
        poses: ``(T, 4, 4)`` camera-to-world matrices.
        height: raymap height in latent pixels.
        width: raymap width in latent pixels.
        use_plucker: return ``[direction, moment]`` if true, otherwise
            ``[origin, direction]``.
    """

    if intrinsics.ndim != 2 or intrinsics.shape[-1] != 4:
        raise ValueError(f"Sana-WM intrinsics must have shape (T, 4), got {tuple(intrinsics.shape)}.")
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Sana-WM poses must have shape (T, 4, 4), got {tuple(poses.shape)}.")
    if intrinsics.shape[0] != poses.shape[0]:
        raise ValueError("Sana-WM intrinsics and poses must have the same temporal length.")

    frames = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x_grid = x_grid[None].expand(frames, -1, -1)
    y_grid = y_grid[None].expand(frames, -1, -1)

    fx = intrinsics[:, 0].view(frames, 1, 1)
    fy = intrinsics[:, 1].view(frames, 1, 1)
    cx = intrinsics[:, 2].view(frames, 1, 1)
    cy = intrinsics[:, 3].view(frames, 1, 1)
    dirs_cam = torch.stack([(x_grid - cx) / fx, (y_grid - cy) / fy, torch.ones_like(x_grid)], dim=-1)

    rotation = poses[:, :3, :3]
    translation = poses[:, :3, 3]
    dirs_world = torch.einsum("tij,thwj->thwi", rotation, dirs_cam)
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True).clamp_min(1e-12)
    origins = translation.view(frames, 1, 1, 3).expand_as(dirs_world)

    if use_plucker:
        moments = torch.cross(origins, dirs_world, dim=-1)
        return torch.cat([dirs_world, moments], dim=-1)
    return torch.cat([origins, dirs_world], dim=-1)


def intrinsics_to_vec4_array(intrinsics: Any, *, num_frames: int, height: int, width: int) -> np.ndarray:
    """Normalize the intrinsics payload to an ``(F, 4)`` ``[fx, fy, cx, cy]`` array.

    Accepts either ``None`` (derive from the output resolution) or the
    ``{fx, fy, cx, cy}`` mapping — the only form the request contract exposes.
    """

    if intrinsics is None:
        focal = float(max(height, width))
        vec = np.array([focal, focal, width / 2.0, height / 2.0], dtype=np.float32)
        return np.broadcast_to(vec, (num_frames, 4)).copy()

    if isinstance(intrinsics, dict):
        vec = np.array(
            [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
            dtype=np.float32,
        )
        return np.broadcast_to(vec, (num_frames, 4)).copy()

    raise ValueError("Sana-WM intrinsics must be a {fx, fy, cx, cy} mapping.")


def _pack_camera_conditions(
    poses: torch.Tensor,
    intrinsics_latent: torch.Tensor,
    *,
    num_frames: int,
    latent_height: int,
    latent_width: int,
    vae_time_stride: int,
) -> dict[str, torch.Tensor]:
    # This yields exactly the latent frame count the VAE produces:
    # ``len(arange(0, n, s)) == ceil(n / s) == (n - 1) // s + 1``.
    time_indices = torch.arange(0, num_frames, vae_time_stride)

    raymap = torch.cat([poses[time_indices].reshape(len(time_indices), -1), intrinsics_latent[time_indices]], dim=-1)

    chunks: list[torch.Tensor] = []
    spatial_raymap_frames: list[torch.Tensor] = []
    chunk_starts = time_indices - (vae_time_stride - 1)
    for start, t_idx in zip(chunk_starts, time_indices):
        start_idx = max(0, int(start))
        end_idx = start_idx + vae_time_stride
        chunk_poses = poses[start_idx:end_idx]
        chunk_intrinsics = intrinsics_latent[start_idx:end_idx]
        if chunk_poses.shape[0] < vae_time_stride:
            pad = vae_time_stride - chunk_poses.shape[0]
            chunk_poses = torch.cat([chunk_poses, chunk_poses[-1:].repeat(pad, 1, 1)], dim=0)
            chunk_intrinsics = torch.cat([chunk_intrinsics, chunk_intrinsics[-1:].repeat(pad, 1)], dim=0)
        plucker = compute_raymap(chunk_intrinsics, chunk_poses, latent_height, latent_width, use_plucker=True)
        chunks.append(plucker.permute(0, 3, 1, 2).reshape(-1, latent_height, latent_width))

        # Per-frame spatial ray-direction map for raymap_embedder: take the
        # representative frame in each chunk (the anchor at t_idx) and use only
        # the direction channels (last 3 of 6 in the origin+direction layout).
        t = int(t_idx)
        od = compute_raymap(
            intrinsics_latent[t : t + 1],
            poses[t : t + 1],
            latent_height,
            latent_width,
            use_plucker=False,
        )  # [1, H, W, 6]
        spatial_raymap_frames.append(od[0, :, :, 3:])  # [H, W, 3] — direction only

    chunk_plucker = torch.stack(chunks).permute(1, 0, 2, 3)
    # spatial_raymap: [3, F_latent, H, W] — consumed by raymap_embedder
    spatial_raymap = torch.stack(spatial_raymap_frames).permute(3, 0, 1, 2)
    return {"raymap": raymap, "chunk_plucker": chunk_plucker, "spatial_raymap": spatial_raymap}


def build_plucker_condition(
    condition: SanaWmCameraCondition,
    *,
    vae_stride: tuple[int, int, int] | list[int] = SANA_WM_DEFAULT_VAE_STRIDE,
) -> dict[str, torch.Tensor]:
    """Build native Stage-1 camera tensors from normalized request metadata."""

    if bool(condition.action) == (condition.poses is not None):
        raise ValueError("Sana-WM camera condition requires exactly one of action or poses.")

    if condition.action:
        poses_c2w = action_string_to_c2w(
            condition.action,
            translation_speed=condition.translation_speed,
            rotation_speed_deg=condition.rotation_speed_deg,
            pitch_limit_deg=condition.pitch_limit_deg,
        )
    else:
        poses_c2w = np.asarray(condition.poses, dtype=np.float32)

    if poses_c2w.ndim != 3 or poses_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Sana-WM c2w poses must have shape (F, 4, 4), got {poses_c2w.shape}.")

    num_frames = condition.num_frames or int(poses_c2w.shape[0])
    if poses_c2w.shape[0] < num_frames:
        if condition.action:
            num_frames = int(poses_c2w.shape[0])
        else:
            raise ValueError(f"Sana-WM c2w poses length {poses_c2w.shape[0]} is shorter than num_frames {num_frames}.")
    poses_c2w = poses_c2w[:num_frames]

    intrinsics = intrinsics_to_vec4_array(
        condition.intrinsics,
        num_frames=num_frames,
        height=condition.height,
        width=condition.width,
    )

    vae_time_stride = int(vae_stride[0])
    vae_spatial_stride = int(vae_stride[-1])
    latent_height = condition.height // vae_spatial_stride
    latent_width = condition.width // vae_spatial_stride

    poses = torch.from_numpy(poses_c2w).float()
    first_inv = get_pose_inverse(poses[0:1]).squeeze(0)
    poses_rel = torch.matmul(first_inv, poses[1:])
    poses = torch.cat([torch.eye(4, dtype=poses.dtype).unsqueeze(0), poses_rel], dim=0)

    intrinsics_latent = torch.from_numpy(intrinsics).float()
    intrinsics_latent[:, [0, 2]] *= latent_width / float(condition.width)
    intrinsics_latent[:, [1, 3]] *= latent_height / float(condition.height)

    return _pack_camera_conditions(
        poses,
        intrinsics_latent,
        num_frames=num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        vae_time_stride=vae_time_stride,
    )
