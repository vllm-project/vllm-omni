from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from openai.types import realtime as types


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:24]}"


@dataclass(slots=True)
class ResponseUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class ActiveResponse:
    """Engine request backing the active Realtime response."""

    response_id: str
    request_id: str
    item_id: str | None = None


def _default_config() -> types.RealtimeSessionCreateRequest:
    return types.RealtimeSessionCreateRequest.model_validate(
        {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": None,
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            },
            "output_modalities": ["audio"],
            "max_output_tokens": "inf",
            "truncation": "auto",
        }
    )


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_session_config(
    current: types.RealtimeSessionCreateRequest,
    update: types.RealtimeSessionCreateRequest,
) -> types.RealtimeSessionCreateRequest:
    # Explicit nulls in a partial session.update must overwrite current values.
    base = current.model_dump()
    patch = update.model_dump(exclude_unset=True)
    return types.RealtimeSessionCreateRequest.model_validate(_deep_merge(base, patch))


@dataclass
class AudioFullDuplexSessionState:
    session_id: str = field(default_factory=lambda: _gen_id("sess"))
    created_at: float = field(default_factory=time.time)
    # Session expiration is advertised but not currently enforced.
    expires_at: float = field(default_factory=lambda: time.time() + 1800)

    config: types.RealtimeSessionCreateRequest = field(default_factory=_default_config)

    conversation_id: str = field(default_factory=lambda: _gen_id("conv"))
    items: list[Any] = field(default_factory=list)

    item_duration_ms: dict[str, float] = field(default_factory=dict)
    item_token_ids: dict[str, list[int]] = field(default_factory=dict)
    item_in_progress: dict[str, bool] = field(default_factory=dict)
    pending_truncations_ms: dict[str, int] = field(default_factory=dict)

    input_audio_buffer: bytearray = field(default_factory=bytearray)
    active_response: ActiveResponse | None = None

    def find_item_index(self, item_id: str) -> int | None:
        for i, item in enumerate(self.items):
            if item.id == item_id:
                return i
        return None

    def find_item(self, item_id: str) -> Any | None:
        idx = self.find_item_index(item_id)
        return self.items[idx] if idx is not None else None

    def insert_item(self, item: Any, previous_item_id: str | None = None) -> int:
        if item.id is not None:
            existing_idx = self.find_item_index(item.id)
            if existing_idx is not None:
                raise ValueError(f"Item '{item.id}' already exists")

        if item.id is None:
            item.id = _gen_id("item")
        if item.object is None:
            item.object = "realtime.item"
        if item.status is None:
            item.status = "completed"

        if previous_item_id is None:
            pos = len(self.items)
        elif previous_item_id == "root":
            pos = 0
        else:
            idx = self.find_item_index(previous_item_id)
            if idx is None:
                raise ValueError(f"previous_item_id '{previous_item_id}' not found")
            pos = idx + 1

        self.items.insert(pos, item)
        return pos

    def replace_item(self, item: Any) -> int:
        if item.id is None:
            raise ValueError("Replacement item must have an id")
        idx = self.find_item_index(item.id)
        if idx is None:
            raise ValueError(f"Item '{item.id}' not found")
        if item.object is None:
            item.object = "realtime.item"
        if item.status is None:
            item.status = "completed"
        self.items[idx] = item
        return idx

    def _clear_item_metadata(self, item_id: str) -> None:
        self.item_duration_ms.pop(item_id, None)
        self.item_token_ids.pop(item_id, None)
        self.item_in_progress.pop(item_id, None)
        self.pending_truncations_ms.pop(item_id, None)

    def remove_item(self, item_id: str) -> Any | None:
        idx = self.find_item_index(item_id)
        if idx is None:
            return None
        item = self.items.pop(idx)
        if item.id:
            self._clear_item_metadata(item.id)
        return item
