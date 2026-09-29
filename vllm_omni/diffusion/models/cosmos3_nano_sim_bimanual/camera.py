# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Standalone camera trajectory frontend for Bimanual inference.

Uses OpenCV local transforms and column-based rot6d ordering. Translations move
0.1 units per step; rotations turn 0.5 degrees per step. Single-axis rotations
are evaluated with NumPy.
Invalid commands are rejected rather than silently omitted.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

_COMMAND = re.compile(r"([a-z]+)-([1-9][0-9]*)(?:-([0-9]+(?:\.[0-9]+)?))?\Z")
_TRANSLATIONS = {"w": (2, 1), "s": (2, -1), "a": (0, -1), "d": (0, 1), "u": (1, -1), "n": (1, 1)}
_ROTATIONS = {"up": (0, 1), "down": (0, -1), "left": (1, -1), "right": (1, 1), "cw": (2, 1), "ccw": (2, -1)}


def _rotation(axis: int, angle: float) -> np.ndarray:
    matrix = np.eye(3)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    matrix[i, i] = matrix[j, j] = np.cos(angle)
    matrix[i, j], matrix[j, i] = -np.sin(angle), np.sin(angle)
    return matrix


def parse_camera_string(spec: str) -> np.ndarray:
    """Return initial identity plus one pose per commanded frame."""
    current = np.eye(4)
    poses = [current.copy()]
    for token in spec.split(","):
        match = _COMMAND.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"Invalid camera command {token!r}; expected command-positive_frames[-radius].")
        command, frames_text, radius_text = match.groups()
        frames = int(frames_text)
        if radius_text is not None and command != "orbit":
            raise ValueError("Only orbit accepts a radius.")
        delta = np.eye(4)
        if command in _TRANSLATIONS:
            axis, direction = _TRANSLATIONS[command]
            delta[axis, 3] = direction * 0.1
        elif command in _ROTATIONS:
            axis, direction = _ROTATIONS[command]
            delta[:3, :3] = _rotation(axis, direction * np.deg2rad(0.5))
        elif command == "pano":
            delta[:3, :3] = _rotation(1, 2 * np.pi / frames)
        elif command == "orbit":
            radius = float(radius_text) if radius_text is not None else 1.0
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Orbit radius must be finite and positive.")
            angle = -2 * np.pi / frames
            delta[:3, :3] = _rotation(1, angle)
            delta[:3, 3] = [-radius * np.sin(angle), 0, radius * (1 - np.cos(angle))]
        elif command != "stay":
            raise ValueError(f"Unknown camera command {command!r}.")
        for _ in range(frames):
            current = current @ delta
            poses.append(current.copy())
    return np.stack(poses)


def resolve_camera_poses(spec: str, target_frames: int) -> np.ndarray:
    """Resolve and align absolute c2w poses before computing relative actions."""
    if isinstance(target_frames, bool) or not isinstance(target_frames, int) or target_frames < 2:
        raise ValueError("Camera trajectories require at least two target frames.")
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("camera_trajectory must be a command string or absolute pose JSON path.")
    if spec.strip().endswith(".json"):
        path = Path(spec.strip())
        if not path.is_absolute():
            raise ValueError("Camera pose JSON paths must be absolute.")
        poses = np.asarray(json.loads(path.read_text(encoding="utf-8")))
        if poses.dtype.kind not in "iuf":
            raise ValueError("Camera poses must contain real numeric values.")
        poses = poses.astype(np.float64)
    else:
        poses = parse_camera_string(spec)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0 or not np.isfinite(poses).all():
        raise ValueError("Camera poses must be a non-empty finite [T, 4, 4] array.")
    if not np.allclose(poses[0], np.eye(4), rtol=0, atol=1e-6):
        raise ValueError("OpenCV c2w camera poses must be anchored at T_0 = I.")
    rotation = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-6)
        or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), rtol=0, atol=1e-4)
        or not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-4)
    ):
        raise ValueError("Camera poses must contain rigid, proper rotations and homogeneous transforms.")
    if len(poses) < target_frames:
        poses = np.concatenate([poses, np.repeat(poses[-1:], target_frames - len(poses), axis=0)])
    return poses[:target_frames]


def camera_poses_to_actions(poses: np.ndarray, pose_convention: str) -> np.ndarray:
    """Encode each target using the preceding frame or 16-target chunk anchor."""
    if pose_convention not in ("backward_framewise", "backward_chunk_anchored_16f"):
        raise ValueError("Unsupported camera pose convention; absolute conditioning is not supported.")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("Expected at least two [4, 4] camera poses.")
    inverse = np.linalg.inv(poses)
    actions = []
    for index in range(len(poses) - 1):
        anchor = index if pose_convention == "backward_framewise" else 16 * (index // 16)
        delta = (inverse[anchor] @ poses[index + 1]).astype(np.float32)
        rot6d = delta[:3, :2].T.reshape(6)
        actions.append(np.concatenate([delta[:3, 3], rot6d]))
    return np.stack(actions).astype(np.float32)
