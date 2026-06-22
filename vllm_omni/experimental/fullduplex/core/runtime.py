# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.adapter import DuplexAdapter
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexState

Emit = Callable[[dict], Awaitable[None]]


class DuplexRuntime:
    def __init__(self, session: DuplexSession, adapter: DuplexAdapter) -> None:
        self.session = session
        self.adapter = adapter
        self._capabilities = adapter.capabilities()

    async def run(self, inputs: AsyncIterator[dict], emit: Emit) -> None:
        task: asyncio.Task | None = None
        async for event in inputs:
            etype = event.get("type")
            if etype == ev.INPUT_APPEND:
                modality = event.get("modality", "")
                if modality not in self._capabilities.input_modalities:
                    await emit(ev.error(f"unsupported input modality: {modality}"))
                    continue
                await self.adapter.on_input(self.session, modality, event.get("data"))
                self.session.state = DuplexState.LISTENING
                if self.session.config.proactive and self.adapter.should_respond(self.session):
                    task = await self._start_response(task, emit)
            elif etype in (ev.INPUT_COMMIT, ev.RESPONSE_CREATE):
                if self.adapter.should_respond(self.session):
                    task = await self._start_response(task, emit)
            elif etype == ev.RESPONSE_CANCEL:
                await self._barge_in()
                await self._cancel(task)
                task = None
                await emit(ev.cancelled(self.session.response_index))
            elif etype == ev.PLAYBACK_ACK:
                await self.adapter.on_playback_ack(self.session, int(event.get("cursor", 0)))
            elif etype == ev.CLOSE:
                break
        if task is not None and not task.done():
            await task
        self.session.state = DuplexState.CLOSED

    async def _start_response(self, prev: asyncio.Task | None, emit: Emit) -> asyncio.Task:
        # A new response supersedes any in-flight one. Mark the old response stale and
        # cancel it (barge-in) instead of awaiting its full stream — otherwise the input
        # loop would block until the old response finishes and could not service a
        # cancel / barge-in event in time, breaking the full-duplex contract.
        if prev is not None and not prev.done():
            await self._barge_in()
            await self._cancel(prev)
        return asyncio.create_task(self._respond(emit))

    @staticmethod
    async def _cancel(task: asyncio.Task | None) -> None:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _respond(self, emit: Emit) -> None:
        response_index, epoch = self.session.begin_response()
        await emit(ev.created(response_index))
        try:
            async for chunk in self.adapter.respond(self.session):
                if self.session.is_stale(epoch):
                    return
                await emit(ev.delta(response_index, chunk.modality, chunk.data))
        except Exception as err:
            await emit(ev.error(f"response failed: {err}"))
        finally:
            if not self.session.is_stale(epoch):
                await emit(ev.done(response_index))
                self.session.end_response()

    async def _barge_in(self) -> None:
        self.session.barge_in()
        await self.adapter.on_barge_in(self.session)
