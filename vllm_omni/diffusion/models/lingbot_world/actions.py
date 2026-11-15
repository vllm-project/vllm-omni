# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keyboard camera controls for LingBot World 2.0."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeAlias

import numpy as np
import torch

from vllm_omni.diffusion.models.lingbot_world.camera import (
    CameraTrajectory,
)

LINGBOT_CONTROLLER_TRANSLATION_UNIT = 0.05
_ACTION_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")
_VALID_ACTIONS = frozenset(_ACTION_ORDER)
_REFERENCE_HEIGHT = 480
_REFERENCE_WIDTH = 832


def _normalize_actions(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of LingBot action keys.")
    actions: set[str] = set()
    for item in value:
        if not isinstance(item, str) or item.lower() not in _VALID_ACTIONS:
            raise ValueError(f"{field} supports only W/A/S/D/I/J/K/L action keys.")
        actions.add(item.lower())
    return tuple(action for action in _ACTION_ORDER if action in actions)


#: Key states for one latent frame each, e.g. ``(("w",), ("w", "j"), ())`` for
#: the three latent frames of one AR block.
LingBotCameraActionFrames: TypeAlias = tuple[tuple[str, ...], ...]

#: One :data:`LingBotCameraActionFrames` per generated chunk, carried on the
#: request for the whole rollout.
LingBotCameraActionScript: TypeAlias = tuple[LingBotCameraActionFrames, ...]


def as_camera_action_script(value: Iterable[Iterable[Iterable[str]]]) -> LingBotCameraActionScript:
    """Restore the tuple form of a whole request's script.

    ``sampling_params.extra_args`` round-trips through JSON, so an already
    validated script comes back as lists; this rebuilds the hashable tuple form
    without re-running validation.
    """
    return tuple(tuple(tuple(frame) for frame in chunk) for chunk in value)


def _normalize_frames(value: object, *, field: str) -> LingBotCameraActionFrames:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of per-frame action lists.")
    return tuple(_normalize_actions(actions, field=f"{field}[{index}]") for index, actions in enumerate(value))


def parse_lingbot_camera_action_script(
    script: object,
    *,
    frames_per_chunk: int,
) -> LingBotCameraActionScript:
    """Validate a request-scoped list of per-chunk camera action frames."""

    if not isinstance(script, Sequence) or isinstance(script, (str, bytes)):
        raise ValueError("camera_action_script must be a sequence of per-chunk action lists.")
    if not script:
        raise ValueError("camera_action_script must contain at least one chunk.")
    chunks: list[LingBotCameraActionFrames] = []
    for index, chunk in enumerate(script):
        frames = _normalize_frames(chunk, field=f"camera_action_script[{index}]")
        if len(frames) != frames_per_chunk:
            raise ValueError(
                "camera_action_script chunks must contain exactly "
                f"{frames_per_chunk} per-latent-frame action lists; "
                f"chunk {index} has {len(frames)}."
            )
        chunks.append(frames)
    return tuple(chunks)


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    if axis == "x":
        return np.array(
            ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
            dtype=np.float64,
        )
    return np.array(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def integrate_lingbot_camera_actions(
    frames: Sequence[Sequence[str]],
    *,
    width: int,
    height: int,
    initial_pose: torch.Tensor | None = None,
    initial_pitch: float = 0.0,
) -> tuple[CameraTrajectory, float]:
    """Convert latent-frame WASD/IJKL controls to cumulative C2W poses.

    Calibration and motion constants intentionally match SGLang's LingBot
    adapter: movement 0.05 (LINGBOT_CONTROLLER_TRANSLATION_UNIT), pitch 4 degrees, yaw 6 degrees, pitch clamp 85
    degrees, and target-resolution focal lengths of 500 pixels.
    """

    normalized_frames = tuple(
        _normalize_actions(actions, field=f"camera action frames[{index}]") for index, actions in enumerate(frames)
    )
    if not normalized_frames:
        raise ValueError("camera action frames must not be empty.")
    if width <= 0 or height <= 0:
        raise ValueError("camera action resolution must be positive.")

    if initial_pose is None:
        current_pose = np.eye(4, dtype=np.float64)
    else:
        if initial_pose.shape != (4, 4) or not torch.isfinite(initial_pose).all():
            raise ValueError("initial_pose must be one finite 4x4 camera-to-world matrix.")
        current_pose = initial_pose.detach().cpu().double().numpy().copy()
    current_pitch = float(initial_pitch)
    pitch_limit = math.radians(85.0)
    poses: list[np.ndarray] = []

    for actions in normalized_frames:
        rotation = current_pose[:3, :3]
        translation = current_pose[:3, 3]
        pitch_delta = math.radians(4.0) * (("i" in actions) - ("k" in actions))
        if not -pitch_limit <= current_pitch + pitch_delta <= pitch_limit:
            pitch_delta = 0.0
        else:
            current_pitch += pitch_delta
        yaw_delta = math.radians(6.0) * (("l" in actions) - ("j" in actions))
        new_rotation = _rotation_matrix("y", yaw_delta) @ rotation @ _rotation_matrix("x", pitch_delta)

        forward = np.array((new_rotation[0, 2], 0.0, new_rotation[2, 2]))
        right = np.array((new_rotation[0, 0], 0.0, new_rotation[2, 0]))
        forward_norm = np.linalg.norm(forward)
        right_norm = np.linalg.norm(right)
        if forward_norm > 0:
            forward /= forward_norm + 1e-6
        if right_norm > 0:
            right /= right_norm + 1e-6

        movement = np.zeros(3, dtype=np.float64)
        movement += forward * LINGBOT_CONTROLLER_TRANSLATION_UNIT * (("w" in actions) - ("s" in actions))
        movement += right * LINGBOT_CONTROLLER_TRANSLATION_UNIT * (("d" in actions) - ("a" in actions))
        # Keep the cumulative controller pose in FP64 between chunks, matching
        # SGLang's NumPy integration and avoiding long-session drift.
        current_pose = np.eye(4, dtype=np.float64)
        current_pose[:3, :3] = new_rotation
        current_pose[:3, 3] = translation + movement
        poses.append(current_pose)

    trajectory = camera_trajectory_from_absolute_pose(torch.from_numpy(np.stack(poses)), width=width, height=height)
    return trajectory, current_pitch


def camera_trajectory_from_absolute_pose(
    poses: torch.Tensor,
    *,
    width: int,
    height: int,
) -> CameraTrajectory:
    if width <= 0 or height <= 0:
        raise ValueError("camera action resolution must be positive.")
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"absolute camera poses must have shape [frames, 4, 4], got {tuple(poses.shape)}.")
    if poses.shape[0] == 0:
        raise ValueError("absolute camera poses must not be empty.")
    # build_plucker_embedding expects intrinsics in the 832x480 reference
    # coordinate system. These values become SGLang's [500, 500, W/2, H/2]
    # after that function scales them to the requested resolution.
    reference_intrinsics = (
        500.0 * _REFERENCE_WIDTH / width,
        500.0 * _REFERENCE_HEIGHT / height,
        _REFERENCE_WIDTH / 2,
        _REFERENCE_HEIGHT / 2,
    )
    trajectory = CameraTrajectory(
        poses=poses,
        intrinsics=torch.tensor(reference_intrinsics, dtype=torch.float32).repeat(len(poses), 1),
    )
    return trajectory
