# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FIFO admission for complete requests sharing one AR/DiT weight group."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager

from vllm_omni.distributed.weight_sharing.runtime import SharingRuntime
from vllm_omni.engine.messages import (
    AddCompanionRequestMessage,
    EngineQueueMessage,
    NextStageInputMessage,
    StageSubmissionMessage,
)


class SharedRequestQueue:
    """Keep one logical request active until terminal cleanup and any abort drain.

    All methods run on the orchestrator loop. Waiting here leaves its request
    handler free to receive cancellations, health queries and shutdown messages.
    """

    def __init__(self, sharing: SharingRuntime) -> None:
        self.sharing = sharing
        self.pending: dict[str, list[EngineQueueMessage]] = {}
        self.active: str | None = None
        self._submitting: list[EngineQueueMessage] | None = None
        self._wake = asyncio.Event()
        self._finished = asyncio.Event()
        self._aborted = False
        self._submissions = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @asynccontextmanager
    async def submission(self, request_id: str) -> AsyncIterator[bool]:
        """Keep cancellation draining behind any already-started stage submit."""
        if request_id != self.active or self._finished.is_set():
            yield False
            return
        self._submissions += 1
        self._idle.clear()
        try:
            yield True
        finally:
            self._submissions -= 1
            if not self._submissions:
                self._idle.set()

    def enqueue(self, message: EngineQueueMessage) -> bool:
        """Queue new requests and their deferred companions; leave controls alone."""
        if isinstance(message, NextStageInputMessage) or (
            isinstance(message, StageSubmissionMessage) and message.type == "add_request"
        ):
            request_id = message.request_id
            if request_id == self.active or request_id in self.pending:
                raise ValueError(f"duplicate shared request ID: {request_id}")
            self.pending[request_id] = [message]
            self._wake.set()
            return True
        if isinstance(message, AddCompanionRequestMessage):
            messages = self.pending.get(message.parent_id)
            if message.parent_id == self.active:
                messages = self._submitting
            if messages is not None:
                messages.append(message)
                return True
        return False

    def discard(self, request_ids: Iterable[str]) -> list[EngineQueueMessage]:
        """Remove queued requests before asynchronous cancellation can yield."""
        return [message for request_id in request_ids for message in self.pending.pop(request_id, ())]

    def finish(self, request_ids: Iterable[str], *, aborted: bool) -> None:
        """Signal logical cleanup; an abort still requires a device drain."""
        if self.active is not None and self.active in request_ids:
            self._aborted |= aborted
            self._finished.set()

    async def run(
        self,
        dispatch: Callable[[EngineQueueMessage], Awaitable[bool]],
        stop: asyncio.Event,
    ) -> None:
        """Run admission until shutdown, cancelling waits without releasing work."""
        consumer = asyncio.create_task(self._consume(dispatch, stop))
        shutdown = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait({consumer, shutdown}, return_when=asyncio.FIRST_COMPLETED)
            if consumer in done:
                await consumer
        finally:
            consumer.cancel()
            shutdown.cancel()
            await asyncio.gather(consumer, shutdown, return_exceptions=True)

    async def _consume(
        self,
        dispatch: Callable[[EngineQueueMessage], Awaitable[bool]],
        stop: asyncio.Event,
    ) -> None:
        """Dispatch FIFO requests, propagating failures without releasing the slot."""
        try:
            while not stop.is_set():
                await self._wake.wait()
                self._wake.clear()
                while self.pending and not stop.is_set():
                    if self.sharing.state != "READY":
                        raise RuntimeError("sharing group is unavailable; stopping request admission")
                    self.active = next(iter(self.pending))
                    messages = self.pending.pop(self.active)
                    self._submitting = messages
                    self._finished.clear()
                    self._aborted = False
                    while messages and not self._finished.is_set():
                        await dispatch(messages.pop(0))
                    self._submitting = None
                    await self._finished.wait()
                    await self._idle.wait()
                    if stop.is_set():
                        return
                    if self._aborted:
                        await self.sharing.drain_requests()
                    self.active = None
        except Exception as exc:
            self.sharing.fail(f"shared request admission: {exc}")
            raise
