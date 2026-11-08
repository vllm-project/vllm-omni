# SPDX-License-Identifier: Apache-2.0
"""Typed request boundary shared by AR-Diffusion models and serving APIs.

The transport layer owns event ordering and snapshots. Model adapters consume
the resulting immutable tick without learning about WebSocket messages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

AR_DIFFUSION_TICK_KEY = "ar_diffusion_tick"


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _freeze_transport_value(value: Any, *, path: str) -> Any:
    """Copy JSON-like transport data into recursively immutable containers."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} mapping keys must be strings.")
            frozen[key] = _freeze_transport_value(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_transport_value(item, path=f"{path}[]") for item in value)
    raise ValueError(
        f"{path} must contain only transport-safe mappings, sequences, and scalar values; got {type(value).__name__}."
    )


def _thaw_transport_value(value: Any) -> Any:
    """Return a detached JSON-serializable copy of an immutable snapshot."""
    if isinstance(value, Mapping):
        return {key: _thaw_transport_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_transport_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ARDiffusionControlInput:
    """One model-defined control track in a chunk-boundary snapshot."""

    track: str
    schema: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        _non_empty_string(self.track, field="control.track")
        _non_empty_string(self.schema, field="control.schema")
        if not isinstance(self.data, Mapping):
            raise ValueError("control.data must be a mapping.")
        object.__setattr__(
            self,
            "data",
            _freeze_transport_value(self.data, path=f"control[{self.track!r}].data"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "schema": self.schema,
            "data": _thaw_transport_value(self.data),
        }

    @classmethod
    def from_dict(cls, value: object) -> ARDiffusionControlInput:
        if not isinstance(value, Mapping):
            raise ValueError("Each control must be a mapping.")
        return cls(
            track=value.get("track"),  # type: ignore[arg-type]
            schema=value.get("schema"),  # type: ignore[arg-type]
            data=value.get("data"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ARDiffusionTickRequest:
    """A single atomic AR-block computation.

    ``request_id`` identifies this computation, ``session_id`` identifies the
    persistent rollout, and ``applied_event_ids`` records which accepted
    interaction events are represented by this immutable snapshot.
    """

    session_id: str
    request_id: str
    chunk_index: int
    applied_event_ids: tuple[int, ...] = ()
    prompt: str | None = None
    controls: tuple[ARDiffusionControlInput, ...] = ()
    reset: bool = False
    close_session: bool = False

    def __post_init__(self) -> None:
        _non_empty_string(self.session_id, field="session_id")
        _non_empty_string(self.request_id, field="request_id")
        _non_negative_int(self.chunk_index, field="chunk_index")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string or None.")
        if not isinstance(self.reset, bool):
            raise ValueError("reset must be a boolean.")
        if not isinstance(self.close_session, bool):
            raise ValueError("close_session must be a boolean.")

        event_ids = tuple(
            _non_negative_int(event_id, field="applied_event_ids item") for event_id in self.applied_event_ids
        )
        if event_ids != tuple(sorted(set(event_ids))):
            raise ValueError("applied_event_ids must be unique and strictly increasing.")
        controls = tuple(self.controls)
        if any(not isinstance(control, ARDiffusionControlInput) for control in controls):
            raise ValueError("controls must contain ARDiffusionControlInput values.")
        object.__setattr__(self, "applied_event_ids", event_ids)
        object.__setattr__(self, "controls", controls)

    def to_extra_args(self) -> dict[str, Any]:
        """Serialize the tick under one namespaced ``extra_args`` key."""
        return {
            AR_DIFFUSION_TICK_KEY: {
                "session_id": self.session_id,
                "request_id": self.request_id,
                "chunk_index": self.chunk_index,
                "applied_event_ids": list(self.applied_event_ids),
                "prompt": self.prompt,
                "controls": [control.to_dict() for control in self.controls],
                "reset": self.reset,
                "close_session": self.close_session,
            }
        }

    @classmethod
    def from_extra_args(
        cls,
        extra_args: Mapping[str, Any] | None,
    ) -> ARDiffusionTickRequest | None:
        """Parse a typed tick, returning ``None`` for legacy AR requests."""
        if extra_args is None or AR_DIFFUSION_TICK_KEY not in extra_args:
            return None
        value = extra_args[AR_DIFFUSION_TICK_KEY]
        if not isinstance(value, Mapping):
            raise ValueError(f"{AR_DIFFUSION_TICK_KEY} must be a mapping.")
        raw_event_ids = value.get("applied_event_ids", ())
        raw_controls = value.get("controls", ())
        if not isinstance(raw_event_ids, Sequence) or isinstance(raw_event_ids, (str, bytes)):
            raise ValueError("applied_event_ids must be a sequence.")
        if not isinstance(raw_controls, Sequence) or isinstance(raw_controls, (str, bytes)):
            raise ValueError("controls must be a sequence.")
        return cls(
            session_id=value.get("session_id"),  # type: ignore[arg-type]
            request_id=value.get("request_id"),  # type: ignore[arg-type]
            chunk_index=value.get("chunk_index"),  # type: ignore[arg-type]
            applied_event_ids=tuple(raw_event_ids),
            prompt=value.get("prompt"),  # type: ignore[arg-type]
            controls=tuple(ARDiffusionControlInput.from_dict(control) for control in raw_controls),
            reset=value.get("reset", False),  # type: ignore[arg-type]
            close_session=value.get("close_session", False),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ARDiffusionChunkMetadata:
    """Identity metadata returned with one generated AR block."""

    session_id: str
    request_id: str
    chunk_index: int
    applied_event_ids: tuple[int, ...] = ()

    @classmethod
    def from_tick(cls, tick: ARDiffusionTickRequest) -> ARDiffusionChunkMetadata:
        return cls(
            session_id=tick.session_id,
            request_id=tick.request_id,
            chunk_index=tick.chunk_index,
            applied_event_ids=tick.applied_event_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "chunk_index": self.chunk_index,
            "applied_event_ids": list(self.applied_event_ids),
        }
