# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""minWM camera trajectories: ``"a*4,w*8,s*7"`` strings to per-frame camera matrices.

Each step moves the camera 0.08 units or turns it 3 degrees for one latent
frame, in the OpenCV camera frame (+X right, +Y down, +Z forward). The first
frame is the identity pose, so ``N`` steps describe ``N + 1`` latent frames.
Tokens: ``w``/``s`` forward/backward, ``a``/``d`` left/right, ``u``/``dn`` up/down,
``j``/``l`` yaw left/right, ``i``/``k`` pitch up/down.
"""

from __future__ import annotations

import numpy as np
import regex as re
import torch

_TRANSLATION_STEP = 0.08
_ROTATION_STEP = float(np.radians(3.0))
_MOTIONS: dict[str, tuple[str, float]] = {
    "w": ("forward", _TRANSLATION_STEP),
    "s": ("forward", -_TRANSLATION_STEP),
    "d": ("right", _TRANSLATION_STEP),
    "a": ("right", -_TRANSLATION_STEP),
    "u": ("up", _TRANSLATION_STEP),
    "dn": ("up", -_TRANSLATION_STEP),
    "j": ("yaw", -_ROTATION_STEP),
    "l": ("yaw", _ROTATION_STEP),
    "i": ("pitch", _ROTATION_STEP),
    "k": ("pitch", -_ROTATION_STEP),
}
_SEGMENT = re.compile(r"([a-z]+)\*(\d+)")
# Normalized pinhole intrinsics the checkpoint was trained with.
_INTRINSICS = np.array([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def trajectory_viewmats(trajectory: str) -> np.ndarray:
    """Parse a trajectory string into ``[steps + 1, 4, 4]`` float32 world-to-camera matrices."""
    motions: list[tuple[str, float]] = []
    for segment in trajectory.strip().split(","):
        match = _SEGMENT.fullmatch(segment.strip())
        if match is None:
            raise ValueError(f"Cannot parse trajectory segment {segment.strip()!r}; expected e.g. 'w*8'.")
        token, count = match.group(1), int(match.group(2))
        if token not in _MOTIONS:
            raise ValueError(f"Unknown trajectory direction {token!r}; valid: {sorted(_MOTIONS)}.")
        motions.extend([_MOTIONS[token]] * count)

    camera_to_world = np.eye(4)
    poses = [camera_to_world.copy()]
    for kind, amount in motions:
        rotation = camera_to_world[:3, :3]
        if kind == "yaw":
            camera_to_world[:3, :3] = rotation @ _rotation_y(amount)
        elif kind == "pitch":
            camera_to_world[:3, :3] = rotation @ _rotation_x(amount)
        elif kind == "forward":
            camera_to_world[:3, 3] += rotation @ np.array([0.0, 0.0, amount])
        elif kind == "right":
            camera_to_world[:3, 3] += rotation @ np.array([amount, 0.0, 0.0])
        else:
            camera_to_world[:3, 3] += rotation @ np.array([0.0, -amount, 0.0])
        poses.append(camera_to_world.copy())
    return np.stack([np.linalg.inv(pose) for pose in poses]).astype(np.float32)


def trajectory_cameras(trajectory: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 ``[frames, 4, 4]`` view matrices and ``[frames, 3, 3]`` intrinsics of a trajectory."""
    viewmats = torch.from_numpy(trajectory_viewmats(trajectory))
    intrinsics = torch.from_numpy(np.tile(_INTRINSICS, (viewmats.shape[0], 1, 1)))
    return viewmats, intrinsics


def _rotation_x(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_y(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
