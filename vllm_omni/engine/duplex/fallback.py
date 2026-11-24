# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Immutable values exchanged by the engine/API chat-fallback bridge."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

FALLBACK_AUDIO_PAYLOAD_KEY = "_duplex_fallback_audio"


def _freeze(value: object) -> object:
    """Copy JSON-shaped values into containers that cannot be mutated in place."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return deepcopy(value)


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by the annotation
        raise TypeError("fallback payload must be a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class DuplexFallbackRequest:
    """Snapshot of one engine-owned response to execute in the API process."""

    session_id: str
    request_id: str
    response_id: str
    epoch: int
    history: tuple[Mapping[str, object], ...]
    response_config: Mapping[str, object]
    input_payload: Mapping[str, object] | None
    policy_messages: tuple[Mapping[str, object], ...]
    #: Scheduling audio may aggregate several individually committed items.
    #: Their per-item payloads already represent it in the history snapshot.
    input_in_history: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", tuple(_freeze_mapping(item) for item in self.history))
        object.__setattr__(self, "response_config", _freeze_mapping(self.response_config))
        if self.input_payload is not None:
            object.__setattr__(self, "input_payload", _freeze_mapping(self.input_payload))
        object.__setattr__(self, "policy_messages", tuple(_freeze_mapping(item) for item in self.policy_messages))


__all__ = ["DuplexFallbackRequest", "FALLBACK_AUDIO_PAYLOAD_KEY"]
