# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, single-consumer delivery across the engine and caller threads.

Only queued, unsequenced audio can be removed. The producer still rejects
new stale model output; this buffer does not retain a growing response-id
tombstone table. It tracks just the most recently dequeued event until the
next ``get()`` so a consumer can recheck it immediately before delivery.

Limits cover queued events, with a separate small termination reserve and
one final session-closure notification independent of that reserve. A
consumer may additionally hold one dequeued event, at most one queue budget
in size. Already journaled or delivered events are outside this buffer.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from vllm_omni.engine.duplex.events import AudioDelta, DuplexEvent, ErrorEvent, ResponseDone, SessionClosed

_MAX_TERMINAL_BYTES = 4096


class DuplexOutputOverflowError(RuntimeError):
    """One session exceeded its pending output budget; valid output was not dropped."""


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    event: DuplexEvent
    size: int
    reserved: bool


class DuplexOutputBuffer:
    """Nonblocking producer, one async consumer, and response-scoped audio removal.

    ``get()`` releases tracking of its previous result. The consumer must
    finish delivering that result before requesting another. ``guard()``
    makes the final validity check and synchronous journal recording atomic
    with invalidation; never await while holding that guard.
    """

    def __init__(
        self,
        *,
        max_bytes: int,
        max_events: int,
        reserve_bytes: int = 64 * 1024,
        reserve_events: int = 8,
    ) -> None:
        if min(max_bytes, max_events, reserve_bytes, reserve_events) <= 0:
            raise ValueError("output buffer limits must be positive")
        self._max_bytes = max_bytes
        self._max_events = max_events
        self._reserve_bytes = reserve_bytes
        self._reserve_events = reserve_events
        self._lock = threading.Lock()
        self._pending: deque[_PendingEvent] = deque()
        self._bytes = 0
        self._reserved_bytes = self._reserved_events = 0
        self._held: DuplexEvent | None = None
        self._waiter: tuple[asyncio.AbstractEventLoop, asyncio.Future[None]] | None = None
        self._closed = False
        self._terminal: SessionClosed | None = None

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return self._bytes + self._reserved_bytes

    @property
    def pending_events(self) -> int:
        with self._lock:
            return len(self._pending)

    def put(self, event: DuplexEvent) -> bool:
        """Append without blocking; overflow leaves the queue unchanged.

        Return false for late output after closure. A session terminal closes
        through its own single slot, even when the ordinary reserve is full.
        Error and response terminals can use the finite reserve.
        They retain FIFO order and cannot overtake still-valid media. The
        caller must stop the affected session after ordinary overflow instead
        of repeatedly generating errors until the reserve also overflows.
        """
        if isinstance(event, SessionClosed):
            self.close(event)
            return True
        payload = event.to_realtime()
        audio_size = 0
        if isinstance(event, AudioDelta):
            # The producer has already encoded this as base64 ASCII. Count it
            # directly without serializing the large string or decoding PCM.
            audio_size = len(event.delta)
            payload["delta"] = ""
        size = audio_size + len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        with self._lock:
            if self._closed:
                return False
            reserved = self._bytes + size > self._max_bytes or (
                len(self._pending) - self._reserved_events >= self._max_events
            )
            if reserved and (
                not isinstance(event, ErrorEvent | ResponseDone)
                or self._reserved_bytes + size > self._reserve_bytes
                or self._reserved_events >= self._reserve_events
            ):
                raise DuplexOutputOverflowError("duplex session pending output limit exceeded")
            self._pending.append(_PendingEvent(event=event, size=size, reserved=reserved))
            if reserved:
                self._reserved_bytes += size
                self._reserved_events += 1
            else:
                self._bytes += size
            waiter = self._waiter
            self._waiter = None
        self._notify(waiter)
        return True

    @staticmethod
    def _matches(event: DuplexEvent, response_id: str, through_epoch: int) -> bool:
        return (
            isinstance(event, AudioDelta)
            and event.response_id == response_id
            and event.epoch is not None
            and event.epoch <= through_epoch
        )

    def invalidate(self, response_id: str, through_epoch: int) -> None:
        """Remove queued audio for one accepted cancellation.

        Non-audio events and other responses retain their exact order. An
        already dequeued matching event becomes invalid too. The producer
        must not enqueue further stale events after this call.
        """
        with self._lock:
            kept: deque[_PendingEvent] = deque()
            for pending in self._pending:
                if self._matches(pending.event, response_id, through_epoch):
                    self._release(pending)
                else:
                    kept.append(pending)
            self._pending = kept
            if self._held is not None and self._matches(self._held, response_id, through_epoch):
                self._held = None

    def _is_valid(self, event: DuplexEvent) -> bool:
        return not isinstance(event, AudioDelta) or event is self._held

    def is_valid(self, event: DuplexEvent) -> bool:
        """Check the current dequeued event; use ``guard`` for an atomic handoff."""
        with self._lock:
            return self._is_valid(event)

    @contextmanager
    def guard(self, event: DuplexEvent) -> Iterator[bool]:
        """Hold validity stable during synchronous sequencing; never await here."""
        with self._lock:
            yield self._is_valid(event)

    def _release(self, pending: _PendingEvent) -> None:
        if pending.reserved:
            self._reserved_bytes -= pending.size
            self._reserved_events -= 1
        else:
            self._bytes -= pending.size

    async def get(self) -> DuplexEvent | None:
        """Return the next event, or ``None`` once a closed buffer is drained."""
        loop = asyncio.get_running_loop()
        with self._lock:
            self._held = None
        while True:
            with self._lock:
                if self._pending:
                    pending = self._pending.popleft()
                    self._release(pending)
                    self._held = pending.event
                    return pending.event
                if self._closed:
                    if self._terminal is not None:
                        terminal = self._terminal
                        self._terminal = None
                        return terminal
                    return None
                if self._waiter is not None:
                    raise RuntimeError("duplex output buffer already has a waiting consumer")
                future: asyncio.Future[None] = loop.create_future()
                self._waiter = (loop, future)
            try:
                await future
            finally:
                with self._lock:
                    if self._waiter is not None and self._waiter[1] is future:
                        self._waiter = None

    def close(self, event: SessionClosed | None = None) -> None:
        """Finish once, preserving FIFO output before an optional final notification.

        The single final slot cannot be consumed by errors or response endings.
        Oversized close details are replaced by a compact notification retaining
        the generated session/event identity and terminal type. Duplicate closure
        and late output are harmless, including after a local
        shutdown that already ended consumption without a session event.
        """
        if event is not None:
            size = len(json.dumps(event.to_realtime(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if size > _MAX_TERMINAL_BYTES:
                event = replace(event, reason="close_details_exceed_output_limit", details={})
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._terminal = event
            waiter = self._waiter
            self._waiter = None
        self._notify(waiter)

    @staticmethod
    def _notify(waiter: tuple[asyncio.AbstractEventLoop, asyncio.Future[None]] | None) -> None:
        if waiter is None:
            return
        loop, future = waiter

        def wake() -> None:
            if not future.done():
                future.set_result(None)

        try:
            loop.call_soon_threadsafe(wake)
        except RuntimeError:
            if not loop.is_closed():
                raise
