# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

from openai.types import realtime as types
from pydantic import BaseModel

MAX_HISTORY_BYTES = 64 * 1024 * 1024
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class HistoryLimitError(ValueError):
    """Raised when adding an item would exceed the session history limit."""


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
    terminal_event_sent: bool = False


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


def _merge_config_models(base: _ModelT, update: BaseModel) -> _ModelT:
    """Apply only explicitly set fields from one Pydantic model to another."""
    merged = base.model_copy(deep=True)
    for field_name in update.model_fields_set:
        value = getattr(update, field_name)
        current_value = getattr(merged, field_name, None)
        if isinstance(current_value, BaseModel) and isinstance(value, BaseModel) and type(current_value) is type(value):
            value = _merge_config_models(current_value, value)
        setattr(merged, field_name, value)
    return merged


def merge_session_config(
    current: types.RealtimeSessionCreateRequest,
    update: types.RealtimeSessionCreateRequest,
) -> types.RealtimeSessionCreateRequest:
    # Explicit nulls in a partial session.update must overwrite current values.
    return _merge_config_models(current, update)


@dataclass
class AudioFullDuplexSessionState:
    session_id: str = field(default_factory=lambda: _gen_id("sess"))
    created_at: float = field(default_factory=time.time)
    # Session expiration is advertised but not currently enforced.
    expires_at: float = field(default_factory=lambda: time.time() + 1800)

    config: types.RealtimeSessionCreateRequest = field(default_factory=_default_config)

    conversation_id: str = field(default_factory=lambda: _gen_id("conv"))
    items: list[types.ConversationItem] = field(default_factory=list)
    # Valid model-context cursor states are:
    # - (None, False): include from the first retained item.
    # - (first item ID, False): include from that item onward.
    # - (None, True): the cursor is past the end and the model context is
    #   empty; a later append moves the cursor to the appended item.
    # The combination (item ID, True) is invalid.
    model_context_first_item_id: str | None = None
    model_context_cursor_at_end: bool = False

    item_duration_ms: dict[str, float] = field(default_factory=dict)
    item_token_ids: dict[str, list[int]] = field(default_factory=dict)
    item_in_progress: dict[str, bool] = field(default_factory=dict)
    pending_truncations_ms: dict[str, types.ConversationItemTruncateEvent] = field(default_factory=dict)

    input_audio_buffer: bytearray = field(default_factory=bytearray)
    active_response: ActiveResponse | None = None

    def find_item_index(self, item_id: str) -> int | None:
        for i, item in enumerate(self.items):
            if item.id == item_id:
                return i
        return None

    def find_item(self, item_id: str) -> types.ConversationItem | None:
        idx = self.find_item_index(item_id)
        return self.items[idx] if idx is not None else None

    @staticmethod
    def _jsonable_item(item: types.ConversationItem) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json", exclude_none=True)
        return item

    def _history_size(self, items: list[types.ConversationItem] | None = None) -> int:
        history = self.items if items is None else items
        serialized = json.dumps(
            [self._jsonable_item(item) for item in history],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return len(serialized.encode("utf-8"))

    def _ensure_history_capacity(self, items: list[types.ConversationItem]) -> None:
        if self._history_size(items) > MAX_HISTORY_BYTES:
            raise HistoryLimitError(f"Conversation history exceeds the {MAX_HISTORY_BYTES} byte limit")

    def _model_context_start_index(self) -> int:
        if self.model_context_cursor_at_end:
            return len(self.items)
        if self.model_context_first_item_id is None:
            return 0
        index = self.find_item_index(self.model_context_first_item_id)
        return len(self.items) if index is None else index

    def model_context_items(self) -> list[types.ConversationItem]:
        return self.items[self._model_context_start_index() :]

    def commit_model_context_items(self, items: list[types.ConversationItem]) -> None:
        self.model_context_first_item_id = items[0].id if items else None
        self.model_context_cursor_at_end = not items

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

        candidate = [*self.items]
        candidate.insert(pos, item)
        self._ensure_history_capacity(candidate)
        self.items.insert(pos, item)
        # A past-the-end cursor only advances for an item appended after it.
        # Inserting before the end leaves the cursor past every included item.
        if self.model_context_cursor_at_end and pos == len(self.items) - 1:
            self.model_context_first_item_id = item.id
            self.model_context_cursor_at_end = False
        elif not self.model_context_cursor_at_end and self.model_context_first_item_id is None:
            self.model_context_first_item_id = item.id
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
        candidate = [*self.items]
        candidate[idx] = item
        self._ensure_history_capacity(candidate)
        self.items[idx] = item
        return idx

    def _clear_item_metadata(self, item_id: str) -> None:
        self.item_duration_ms.pop(item_id, None)
        self.item_token_ids.pop(item_id, None)
        self.item_in_progress.pop(item_id, None)
        self.pending_truncations_ms.pop(item_id, None)

    def remove_item(self, item_id: str) -> types.ConversationItem | None:
        idx = self.find_item_index(item_id)
        if idx is None:
            return None
        cursor_index = self._model_context_start_index()
        item = self.items.pop(idx)
        if idx == cursor_index and not self.model_context_cursor_at_end:
            if idx < len(self.items):
                self.model_context_first_item_id = self.items[idx].id
            else:
                self.model_context_first_item_id = None
                self.model_context_cursor_at_end = True
        if item.id:
            self._clear_item_metadata(item.id)
        return item
