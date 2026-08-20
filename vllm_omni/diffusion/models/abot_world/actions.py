# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Realtime keyboard controls for ABot-World."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_omni.experimental.ar_diffusion.session import (
    ARDiffusionPreparedControls,
    ARDiffusionSessionEvent,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
)

ABOT_CAMERA_ACTION_SCHEMA = "abot.camera_actions.v1"
_ACTION_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")
_VALID_ACTIONS = frozenset(_ACTION_ORDER)
_REFERENCE_HEIGHT = 480
_REFERENCE_WIDTH = 832


def _normalize_actions(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of ABot action keys.")
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


def parse_abot_camera_action_frames(
    data: Mapping[str, Any],
    *,
    expected_frames: int,
) -> tuple[tuple[str, ...], ...]:
    """Validate the chunk-sized payload emitted by the session reducer."""

    if data.get("mode") != "frames":
        raise ValueError("abot.camera_actions.v1 model ticks require mode='frames'.")
    frames = _normalize_frames(data.get("frames"), field="camera action frames")
    if len(frames) != expected_frames:
        raise ValueError(
            "abot.camera_actions.v1 must contain exactly one action list per "
            f"latent frame; expected {expected_frames}, got {len(frames)}."
        )
    return frames


@dataclass(frozen=True)
class _ABotCameraReducerState:
    mode: str | None = None
    held_actions: tuple[str, ...] = ()
    script_frames: tuple[tuple[str, ...], ...] = ()


class ABotCameraControlReducer:
    """Sample live state/script controls into one ABot AR block.

    State mode preserves held keys across chunks and guarantees a one-frame
    pulse when a press and release both arrive before the next chunk. Script
    mode is a finite FIFO padded with neutral frames after exhaustion.
    """

    def __init__(self, *, frames_per_block: int = 3) -> None:
        if isinstance(frames_per_block, bool) or not isinstance(frames_per_block, int) or frames_per_block <= 0:
            raise ValueError("frames_per_block must be a positive integer.")
        self._frames_per_block = frames_per_block
        self._state = _ABotCameraReducerState()

    @staticmethod
    def _state_transitions(data: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
        transitions = data.get("transitions")
        if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)) or not transitions:
            raise ValueError("ABot state-mode camera actions require non-empty transitions.")
        normalized: list[tuple[str, ...]] = []
        previous_timestamp: int | None = None
        for index, transition in enumerate(transitions):
            if not isinstance(transition, Mapping):
                raise ValueError("Each ABot camera state transition must be a mapping.")
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
            if not (track == "camera" and control.schema == ABOT_CAMERA_ACTION_SCHEMA)
        }
        state = self._state
        pending_transitions: tuple[tuple[str, ...], ...] = ()

        for event in events:
            for control in event.controls:
                if control.track != "camera" or control.schema != ABOT_CAMERA_ACTION_SCHEMA:
                    controls[control.track] = control
                    if control.track == "camera":
                        state = _ABotCameraReducerState()
                        pending_transitions = ()
                    continue

                mode = control.data.get("mode")
                controls.pop("camera", None)
                if mode == "script":
                    state = _ABotCameraReducerState(
                        mode="script",
                        script_frames=self._script(control.data),
                    )
                    pending_transitions = ()
                elif mode == "state":
                    transitions = self._state_transitions(control.data)
                    if state.mode != "state":
                        state = _ABotCameraReducerState(mode="state")
                    else:
                        state = _ABotCameraReducerState(
                            mode="state",
                            held_actions=state.held_actions,
                        )
                    pending_transitions += transitions
                else:
                    raise ValueError("abot.camera_actions.v1 events require mode='state' or mode='script'.")

        if state.mode == "script":
            frames = state.script_frames[: self._frames_per_block]
            frames += ((),) * (self._frames_per_block - len(frames))
            state = _ABotCameraReducerState(
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
            state = _ABotCameraReducerState(
                mode="state",
                held_actions=final_actions,
            )
        else:
            frames = ()

        if frames:
            controls["camera"] = ARDiffusionControlInput(
                track="camera",
                schema=ABOT_CAMERA_ACTION_SCHEMA,
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
        if not isinstance(prepared.private_state, _ABotCameraReducerState):
            raise TypeError("ABot reducer prepared state has an unexpected type.")
        self._state = prepared.private_state

    def reset(self) -> None:
        self._state = _ABotCameraReducerState()
