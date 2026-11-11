# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keyboard camera actions for ABot-World."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

_ACTION_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")
_VALID_ACTIONS = frozenset(_ACTION_ORDER)


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


def _normalize_state_transitions(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("ABot state-mode camera actions require non-empty transitions.")
    normalized: list[tuple[str, ...]] = []
    previous_timestamp: int | None = None
    for index, transition in enumerate(value):
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


def parse_abot_camera_action_script(
    value: object,
    *,
    frames_per_chunk: int = 3,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Validate a request-scoped action script, one frame list per chunk."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("camera_action_script must be a sequence of chunks.")
    chunks = tuple(
        _normalize_frames(chunk, field=f"camera_action_script[{index}]") for index, chunk in enumerate(value)
    )
    if any(len(chunk) != frames_per_chunk for chunk in chunks):
        raise ValueError(
            f"camera_action_script must contain exactly {frames_per_chunk} latent-frame action lists per chunk."
        )
    return chunks


@dataclass(frozen=True)
class ABotCameraUpdate:
    """One validated mid-generation camera update.

    ``actions`` holds key-state transitions in ``state`` mode and one action
    list per latent frame in ``script`` mode.
    """

    mode: Literal["state", "script"]
    actions: tuple[tuple[str, ...], ...]


def parse_abot_camera_update(payload: Mapping[str, object]) -> ABotCameraUpdate:
    """Validate an ABot ``multi_modal_data.camera`` interaction payload."""

    mode = payload.get("mode")
    if mode == "state":
        if set(payload) != {"mode", "transitions"}:
            raise ValueError("ABot state-mode camera updates take exactly mode and transitions.")
        return ABotCameraUpdate(mode="state", actions=_normalize_state_transitions(payload["transitions"]))
    if mode == "script":
        if set(payload) != {"mode", "frames"}:
            raise ValueError("ABot script-mode camera updates take exactly mode and frames.")
        return ABotCameraUpdate(
            mode="script",
            actions=_normalize_frames(payload["frames"], field="camera action script frames"),
        )
    raise ValueError("ABot camera updates require mode='state' or mode='script'.")
