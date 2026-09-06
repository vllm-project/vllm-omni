# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Realtime keyboard controls for LingBot World 2.0."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.models.lingbot_world.camera import (
    CameraTrajectory,
)
from vllm_omni.experimental.ar_diffusion.session import (
    ARDiffusionPreparedControls,
    ARDiffusionSessionEvent,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
)

LINGBOT_CAMERA_ACTION_SCHEMA = "lingbot.camera_actions.v1"
LINGBOT_CAMERA_TRAJECTORY_SCHEMA = "lingbot.camera_trajectory.v1"
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


def _normalize_frames(value: object, *, field: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of per-frame action lists.")
    return tuple(_normalize_actions(actions, field=f"{field}[{index}]") for index, actions in enumerate(value))


def parse_lingbot_camera_action_frames(
    data: Mapping[str, Any],
    *,
    expected_frames: int,
) -> tuple[tuple[str, ...], ...]:
    """Validate the chunk-sized payload emitted by the session reducer."""

    if data.get("mode") != "frames":
        raise ValueError("lingbot.camera_actions.v1 model ticks require mode='frames'.")
    frames = _normalize_frames(data.get("frames"), field="camera action frames")
    if len(frames) != expected_frames:
        raise ValueError(
            "lingbot.camera_actions.v1 must contain exactly one action list per "
            f"latent frame; expected {expected_frames}, got {len(frames)}."
        )
    return frames


@dataclass(frozen=True)
class _LingBotCameraReducerState:
    mode: str | None = None
    held_actions: tuple[str, ...] = ()
    script_frames: tuple[tuple[str, ...], ...] = ()


class LingBotCameraControlReducer:
    """Sample live state/script controls into one LingBot AR block.

    State mode preserves held keys across chunks and guarantees a one-frame
    pulse when a press and release both arrive before the next chunk. Script
    mode is a finite FIFO padded with neutral frames after exhaustion.
    """

    def __init__(self, *, frames_per_block: int = 3) -> None:
        if isinstance(frames_per_block, bool) or not isinstance(frames_per_block, int) or frames_per_block <= 0:
            raise ValueError("frames_per_block must be a positive integer.")
        self._frames_per_block = frames_per_block
        self._state = _LingBotCameraReducerState()

    @staticmethod
    def _state_transitions(data: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
        transitions = data.get("transitions")
        if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)) or not transitions:
            raise ValueError("LingBot state-mode camera actions require non-empty transitions.")
        normalized: list[tuple[str, ...]] = []
        previous_timestamp: int | None = None
        for index, transition in enumerate(transitions):
            if not isinstance(transition, Mapping):
                raise ValueError("Each LingBot camera state transition must be a mapping.")
            timestamp = transition.get("client_ts_ms")
            if timestamp is not None:
                if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                    raise ValueError("client_ts_ms must be a non-negative integer when provided.")
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError("client_ts_ms values must be monotonic within one event.")
                previous_timestamp = timestamp
            normalized.append(
                _normalize_actions(
                    transition.get("actions"),
                    field=f"camera state transitions[{index}].actions",
                )
            )
        return tuple(normalized)

    @staticmethod
    def _script(data: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
        return _normalize_frames(data.get("frames"), field="camera action script frames")

    def prepare(
        self,
        *,
        current_controls: Mapping[str, ARDiffusionControlInput],
        events: Sequence[ARDiffusionSessionEvent],
        chunk_index: int,
    ) -> ARDiffusionPreparedControls:
        del chunk_index
        controls = {
            track: control
            for track, control in current_controls.items()
            if not (track == "camera" and control.schema == LINGBOT_CAMERA_ACTION_SCHEMA)
        }
        state = self._state
        pending_transitions: tuple[tuple[str, ...], ...] = ()

        for event in events:
            for control in event.controls:
                if control.track != "camera" or control.schema != LINGBOT_CAMERA_ACTION_SCHEMA:
                    controls[control.track] = control
                    if control.track == "camera":
                        state = _LingBotCameraReducerState()
                        pending_transitions = ()
                    continue

                mode = control.data.get("mode")
                controls.pop("camera", None)
                if mode == "script":
                    state = _LingBotCameraReducerState(
                        mode="script",
                        script_frames=self._script(control.data),
                    )
                    pending_transitions = ()
                elif mode == "state":
                    transitions = self._state_transitions(control.data)
                    if state.mode != "state":
                        state = _LingBotCameraReducerState(mode="state")
                    else:
                        state = _LingBotCameraReducerState(
                            mode="state",
                            held_actions=state.held_actions,
                        )
                    pending_transitions += transitions
                else:
                    raise ValueError("lingbot.camera_actions.v1 events require mode='state' or mode='script'.")

        if state.mode == "script":
            frames = state.script_frames[: self._frames_per_block]
            frames += ((),) * (self._frames_per_block - len(frames))
            state = _LingBotCameraReducerState(
                mode="script",
                script_frames=state.script_frames[self._frames_per_block :],
            )
        elif state.mode == "state":
            final_actions = pending_transitions[-1] if pending_transitions else state.held_actions
            pulse = next(
                (actions for actions in reversed(pending_transitions) if actions),
                None,
            )
            if pulse is not None and pulse != final_actions:
                frames = (pulse,) + (final_actions,) * (self._frames_per_block - 1)
            else:
                frames = (final_actions,) * self._frames_per_block
            state = _LingBotCameraReducerState(
                mode="state",
                held_actions=final_actions,
            )
        else:
            frames = ()

        if frames:
            controls["camera"] = ARDiffusionControlInput(
                track="camera",
                schema=LINGBOT_CAMERA_ACTION_SCHEMA,
                data={
                    "mode": "frames",
                    "frames": [list(actions) for actions in frames],
                },
            )
        return ARDiffusionPreparedControls(
            controls=tuple(controls[track] for track in sorted(controls)),
            private_state=state,
        )

    def commit(self, prepared: ARDiffusionPreparedControls) -> None:
        if not isinstance(prepared.private_state, _LingBotCameraReducerState):
            raise TypeError("LingBot reducer prepared state has an unexpected type.")
        self._state = prepared.private_state

    def reset(self) -> None:
        self._state = _LingBotCameraReducerState()


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
    adapter: movement 0.05, pitch 4 degrees, yaw 6 degrees, pitch clamp 85
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
        movement += forward * 0.05 * (("w" in actions) - ("s" in actions))
        movement += right * 0.05 * (("d" in actions) - ("a" in actions))
        current_pose = np.eye(4, dtype=np.float64)
        current_pose[:3, :3] = new_rotation
        current_pose[:3, 3] = translation + movement
        poses.append(current_pose)

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
        # Keep the cumulative controller pose in FP64 between chunks, matching
        # SGLang's NumPy integration and avoiding long-session drift.
        poses=torch.from_numpy(np.stack(poses)),
        intrinsics=torch.tensor(reference_intrinsics, dtype=torch.float32).repeat(len(poses), 1),
    )
    return trajectory, current_pitch
