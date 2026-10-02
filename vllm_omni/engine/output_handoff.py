# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Orchestrator → serving-loop output queue that wakes its consumer once per batch.

Reading a ``janus`` queue through ``sync_q.get`` in an executor thread costs two
cross-thread (GIL) handoffs per batch. Here the first put of a producer loop
turn schedules one flush at the end of that turn, which wakes the consumer loop
with a single ``call_soon_threadsafe``; the consumer then takes the messages
with ``drain_nowait`` on its own loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

import janus

T = TypeVar("T")


class LoopHandoffQueue(janus.Queue[T]):
    """A ``janus.Queue`` that can wake one consumer event loop once per producer loop turn."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self._consumer: tuple[asyncio.AbstractEventLoop, Callable[[], None]] | None = None
        # A wake is scheduled on the consumer loop and has not started a drain yet.
        self._consumer_wake_pending = False
        # A flush is scheduled at the end of the producer's current loop turn.
        self._flush_scheduled = False

    def set_consumer(self, loop: asyncio.AbstractEventLoop, wake: Callable[[], None]) -> None:
        """Call *wake* on *loop* when messages are waiting (called on *loop*)."""
        self._consumer = (loop, wake)
        self._consumer_wake_pending = False

    def clear_consumer(self) -> None:
        self._consumer = None
        self._consumer_wake_pending = False

    def drain_nowait(self, max_items: int) -> list[T]:
        """Take up to *max_items* queued messages in order (called on the consumer loop).

        Rearms the wake first, so a message put while this drains wakes the
        consumer again instead of waiting for the next one.
        """
        self._consumer_wake_pending = False
        batch: list[T] = []
        sync_q = self.sync_q
        try:
            while len(batch) < max_items:
                batch.append(sync_q.get_nowait())
        except janus.SyncQueueEmpty:
            pass
        return batch

    # janus calls this with its mutex held, for sync and async puts alike.
    def _put(self, item: T) -> None:
        super()._put(item)
        if self._consumer is None or self._consumer_wake_pending or self._flush_scheduled:
            return
        self._flush_scheduled = True
        try:
            asyncio.get_running_loop().call_soon(self._flush)
        except RuntimeError:
            # A put from outside any loop (the orchestrator's failure path).
            self._flush()

    def _flush(self) -> None:
        self._flush_scheduled = False
        consumer = self._consumer
        if consumer is None or self._consumer_wake_pending:
            return
        self._consumer_wake_pending = True
        try:
            consumer[0].call_soon_threadsafe(consumer[1])
        except RuntimeError:
            # The consumer loop is closed; nobody is left to wake.
            self._consumer_wake_pending = False


__all__ = ["LoopHandoffQueue"]
