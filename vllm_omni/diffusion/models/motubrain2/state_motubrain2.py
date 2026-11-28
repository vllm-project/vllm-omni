# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker-local history for Motubrain real-time chunking (RTC)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from motubrain2.inference.rtc import TrainRTC


@dataclass
class SessionState:
    last_action: np.ndarray | None = None
    rng_state: torch.Tensor | None = None


class SessionStore:
    """Keep committed RTC and RNG state within one bounded rollout cache."""

    def __init__(self, max_sessions: int = 64):
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer.")
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    def peek(self, session_id: str | None) -> SessionState | None:
        return self._sessions.get(session_id) if session_id is not None else None

    def get(self, session_id: str) -> SessionState:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
        else:
            if len(self._sessions) == self.max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def reset(self, session_id: str | None) -> None:
        if session_id is not None:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)


def validate_rtc_config(config: Mapping[str, Any] | None, action_steps: int) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("rtc must be a mapping.")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("rtc.enabled must be a boolean.")
    if not enabled:
        return None
    prefix = config.get("execute_prefix", 1)
    if type(prefix) is not int or prefix < 0:
        raise ValueError("rtc.execute_prefix must be a nonnegative integer.")
    if prefix >= action_steps:
        raise ValueError("rtc.execute_prefix must be smaller than the action horizon.")
    delay = config.get("delay", action_steps - prefix)
    if type(delay) is not int or delay < 1:
        raise ValueError("rtc.delay must be a positive integer.")
    return {"enabled": True, "execute_prefix": prefix, "delay": delay}


def build_rtc(session: SessionState | None, config: Mapping[str, Any] | None, action_steps: int) -> TrainRTC | None:
    if config is None or session is None or session.last_action is None:
        return None
    prefix = config["execute_prefix"]
    previous = session.last_action
    future = previous[prefix : prefix + min(config["delay"], action_steps)]
    if not len(future):
        return None
    from motubrain2.inference.rtc import TrainRTC

    return TrainRTC(anchor=None, future=future, delay=len(future))


def request_rtc(payload: Mapping[str, Any], wire_dim: int, action_steps: int) -> TrainRTC:
    from motubrain2.inference.rtc import TrainRTC

    if not isinstance(payload, Mapping) or set(payload) - {"anchor", "future", "delay"} or "delay" not in payload:
        raise ValueError("rtc requires a delay and accepts optional anchor and future actions.")
    future = payload.get("future")
    if future is None:
        if payload["delay"] != 0:
            raise ValueError("rtc.future is required for a nonzero delay.")
        future = np.empty((0, wire_dim))
    elif isinstance(future, (list, tuple)) and not len(future):
        # A JSON/MessagePack empty list carries no wire width.
        future = np.empty((0, wire_dim))
    rtc = TrainRTC(anchor=payload.get("anchor"), future=future, delay=payload["delay"])
    rtc.prefix(wire_dim, action_steps)
    return rtc
