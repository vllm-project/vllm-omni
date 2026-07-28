# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocketDisconnect

INPUT_EVENTS = frozenset(
    {
        "input.text.append",
        "input_audio_buffer.append",
        "input.commit",
        "input_audio_buffer.commit",
        "response.create",
    }
)

_INPUT_EVENT_ALIASES = {
    "signal_turn": "turn.signal",
    "close_session": "session.close",
    "audio.playback_ack": "playback.ack",
    "input_text.append": "input.text.append",
    "push_text": "input.text.append",
    "input.audio.append": "input_audio_buffer.append",
    "push_chunk": "input_audio_buffer.append",
}
_WAV_AUDIO_ALIASES = frozenset({"input.audio.append", "push_chunk"})


def normalize_duplex_input_event(event: dict[str, object]) -> dict[str, object]:
    """Normalize compatibility aliases before an event enters the mailbox."""
    event_type = event.get("type")
    if not isinstance(event_type, str):
        return event
    canonical_type = _INPUT_EVENT_ALIASES.get(event_type)
    if canonical_type is None:
        return event
    normalized = dict(event)
    normalized["type"] = canonical_type
    if event_type in _WAV_AUDIO_ALIASES:
        normalized.setdefault("format", "wav")
    return normalized


MODEL_OUTPUT_EVENTS = frozenset(
    {
        "response.created",
        "response.listen",
        "response.speak",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_audio.delta",
        "response.audio.delta",
        "response.output_audio.done",
        "response.audio.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.text.delta",
        "response.text.done",
        "response.message",
        "response.output_item.done",
        "response.content_part.done",
        "response.done",
        "runtime.control",
    }
)
DOMAIN_TERMINAL_EVENTS = frozenset(
    {
        "response.done",
        "response.listen",
        "audio.cancelled",
        "input.cancelled",
        "session.closed",
    }
)


def is_input_event(event_type: object) -> bool:
    return isinstance(event_type, str) and event_type in INPUT_EVENTS


@dataclass(frozen=True, slots=True)
class DuplexAppendTaskMeta:
    epoch: int
    mode: str
    final: bool
    response_bound: bool


@dataclass
class DuplexSessionTasks:
    """Connection-independent task handles for one resumable session."""

    native_append_tasks: dict[asyncio.Task[bool], DuplexAppendTaskMeta] = field(default_factory=dict)
    native_append_tail: asyncio.Task[bool] | None = None
    active_response_task: asyncio.Task[None] | None = None

    def track_append_task(
        self,
        task: asyncio.Task[bool],
        *,
        epoch: int,
        mode: str,
        final: bool,
        response_bound: bool,
    ) -> None:
        self.native_append_tasks[task] = DuplexAppendTaskMeta(epoch, mode, final, response_bound)
        task.add_done_callback(self.native_append_tasks.pop)

    def has_response_bound_append_tasks(self) -> bool:
        return any(meta.response_bound for meta in self.native_append_tasks.values())

    async def cancel_append_tasks(self, timeout_s: float = 0.25, *, response_bound_only: bool = False) -> bool:
        tasks = [
            task for task, meta in self.native_append_tasks.items() if not response_bound_only or meta.response_bound
        ]
        if not tasks:
            return False
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s)
        except TimeoutError:
            pass
        return True


@dataclass
class DuplexWebSocketActor:
    """Own ordered WebSocket I/O queues, but no domain identity."""

    websocket: Any
    current_epoch: Callable[[], int | None] | None = None
    session_closed: Callable[[], bool] | None = None
    output_queue: asyncio.Queue[dict[str, object] | None] = field(default_factory=asyncio.Queue)
    mailbox: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    outbound_protocol: Any | None = None
    tasks: DuplexSessionTasks = field(default_factory=DuplexSessionTasks)
    closing: bool = False
    close_reason: str | None = None
    stale_output_dropped: int = 0
    _queued_input_events: int = 0

    @property
    def native_append_tasks(self) -> dict[asyncio.Task[bool], DuplexAppendTaskMeta]:
        return self.tasks.native_append_tasks

    @property
    def active_response_task(self) -> asyncio.Task[None] | None:
        return self.tasks.active_response_task

    @active_response_task.setter
    def active_response_task(self, task: asyncio.Task[None] | None) -> None:
        self.tasks.active_response_task = task

    @property
    def native_append_tail(self) -> asyncio.Task[bool] | None:
        return self.tasks.native_append_tail

    @native_append_tail.setter
    def native_append_tail(self, task: asyncio.Task[bool] | None) -> None:
        self.tasks.native_append_tail = task

    async def enqueue_event(self, event: dict[str, object]) -> None:
        if is_input_event(event.get("type")):
            self._queued_input_events += 1
        await self.mailbox.put(event)

    async def next_event(self) -> dict[str, object]:
        event = await self.mailbox.get()
        if is_input_event(event.get("type")):
            self._queued_input_events = max(0, self._queued_input_events - 1)
        self.mailbox.task_done()
        return event

    def has_queued_input_events(self) -> bool:
        return self._queued_input_events > 0

    async def send_json(self, payload: dict[str, object]) -> None:
        await self.output_queue.put(payload)

    async def close_writer(self) -> None:
        await self.output_queue.put(None)

    async def writer_loop(self) -> None:
        while True:
            payload = await self.output_queue.get()
            try:
                if payload is None:
                    return
                raw_realtime = payload.pop("_realtime_raw", False) is True
                if not raw_realtime and self._is_stale_model_output(payload):
                    self.stale_output_dropped += 1
                    continue
                try:
                    if raw_realtime:
                        await self.websocket.send_json(payload)
                    elif self.outbound_protocol is not None:
                        for projected in self.outbound_protocol.encode_outbound_event(payload):
                            await self.websocket.send_json(projected)
                    else:
                        await self.websocket.send_json(payload)
                except (WebSocketDisconnect, RuntimeError):
                    return
            finally:
                self.output_queue.task_done()

    def _is_stale_model_output(self, payload: dict[str, object]) -> bool:
        event_type = payload.get("type")
        if event_type in DOMAIN_TERMINAL_EVENTS:
            return False
        if event_type not in MODEL_OUTPUT_EVENTS:
            return False
        if self.closing and event_type not in {"response.listen", "runtime.control"}:
            return True
        expected_epoch = self.current_epoch() if self.current_epoch is not None else None
        if (
            self.session_closed is not None
            and self.session_closed()
            and event_type
            not in {
                "response.listen",
                "runtime.control",
            }
        ):
            return True
        epoch = payload.get("epoch")
        return isinstance(epoch, int) and isinstance(expected_epoch, int) and epoch != expected_epoch

    def track_append_task(
        self,
        task: asyncio.Task[bool],
        *,
        epoch: int,
        mode: str,
        final: bool,
        response_bound: bool,
    ) -> None:
        self.tasks.track_append_task(
            task,
            epoch=epoch,
            mode=mode,
            final=final,
            response_bound=response_bound,
        )

    def has_response_bound_append_tasks(self) -> bool:
        return self.tasks.has_response_bound_append_tasks()

    async def cancel_append_tasks(self, timeout_s: float = 0.25, *, response_bound_only: bool = False) -> bool:
        return await self.tasks.cancel_append_tasks(timeout_s, response_bound_only=response_bound_only)


__all__ = [
    "DOMAIN_TERMINAL_EVENTS",
    "DuplexAppendTaskMeta",
    "DuplexSessionTasks",
    "DuplexWebSocketActor",
    "is_input_event",
]
