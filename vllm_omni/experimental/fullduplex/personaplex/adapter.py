# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex duplex adapter + model-owned lockstep runtime.

PersonaPlex is a *continuous* (frame-clocked, no turn-taking) model: one eternal
``respond()`` that pulls one 80 ms user frame at a time and emits one agent
frame + text token. The shared ``core.DuplexRuntime`` implements the turn
lifecycle (start/cancel per trigger), which does not fit lockstep, so the
lifecycle lives HERE as :class:`PersonaPlexDuplexRuntime` — model policy stays
in the model package, mirroring #3907's model-owned minicpmo45 runtime, and
``core`` remains verbatim upstream.

Stepping happens off the event loop (``asyncio.to_thread``, as Moshi's own
server does) so the input side stays live — that is what makes barge-in work
without any explicit flush.

Depends only on the ``FrameStepper`` protocol, so it is exercised by GPU-free
tests with a stub stepper; ``PersonaPlexEngine`` is the real backend.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.adapter import DuplexAdapter, DuplexCapability, OutputChunk
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime, Emit
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexState
from vllm_omni.experimental.fullduplex.personaplex.engine import FrameStepper
from vllm_omni.experimental.fullduplex.personaplex.session import MAX_PENDING_INPUT, offer_realtime

_CLOSE = object()  # sentinel: drain the inbox and end the eternal response


class PersonaPlexDuplexAdapter(DuplexAdapter):
    """Drive a :class:`FrameStepper` as a continuous full-duplex response."""

    def __init__(
        self,
        stepper: FrameStepper,
        *,
        voice_prompt: str | None = None,
        persona: str | None = None,
    ) -> None:
        self._stepper = stepper
        self._voice_prompt = voice_prompt
        self._persona = persona
        self._frame_size = stepper.frame_size
        # Bounded, drop-oldest: input arriving faster than the stepper can run
        # must shed stale frames (realtime), not grow memory + latency unbounded.
        self._inbox: asyncio.Queue[Any] = asyncio.Queue(maxsize=MAX_PENDING_INPUT)
        self._pending = np.zeros(0, dtype=np.float32)
        self._opened = False

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            input_modalities=frozenset({"audio"}),
            output_modalities=frozenset({"audio", "text"}),
            proactive=True,
        )

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        """Frame incoming PCM into exact 80 ms windows and queue them for the loop."""
        if modality != "audio":
            return
        samples = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, samples]) if self._pending.size else samples
        n = self._frame_size
        while self._pending.shape[0] >= n:
            frame, self._pending = self._pending[:n], self._pending[n:]
            offer_realtime(self._inbox, frame)

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        """The eternal lockstep loop: one user frame in, one agent frame + text out."""
        if not self._opened:
            await asyncio.to_thread(self._stepper.open_session, self._voice_prompt, self._persona)
            self._opened = True
        while True:
            frame = await self._inbox.get()
            if frame is _CLOSE:
                break
            out = await asyncio.to_thread(self._stepper.step, frame)
            if out.audio is not None:
                yield OutputChunk("audio", out.audio)
            if out.text:
                yield OutputChunk("text", out.text)

    async def on_close(self, session: DuplexSession) -> None:
        # Model-owned drain signal (not part of the core DuplexAdapter ABC):
        # PersonaPlexDuplexRuntime calls this on every exit path. Pad + flush a
        # partial final frame (same contract as PersonaPlexSession.flush()) so
        # the tail of the user's audio is stepped, not dropped.
        if self._pending.size:
            pad = self._frame_size - self._pending.shape[0]
            frame = np.concatenate([self._pending, np.zeros(pad, dtype=np.float32)])
            self._pending = np.zeros(0, dtype=np.float32)
            offer_realtime(self._inbox, frame)
        offer_realtime(self._inbox, _CLOSE)

    async def on_barge_in(self, session: DuplexSession) -> None:
        # No-op: Moshi-class models perceive the user continuously, so an interrupt
        # needs no downstream flush — the next frames simply steer the model.
        return


class PersonaPlexDuplexRuntime(DuplexRuntime):
    """Model-owned lockstep lifecycle: start once, never supersede, drain on exit.

    ``core.DuplexRuntime`` runs the turn lifecycle — every trigger starts a new
    response that supersedes and cancels the previous one. A lockstep model has
    exactly ONE response for the whole session: it consumes user frames as they
    arrive and only ends when the input side goes away. Keeping this here (not
    as a flag on the shared runtime) mirrors #3907's model-owned minicpmo45
    runtime: model policy lives in the model package.
    """

    async def run(self, inputs: AsyncIterator[dict], emit: Emit) -> None:
        task: asyncio.Task | None = None
        try:
            async for event in inputs:
                etype = event.get("type")
                if etype == ev.INPUT_APPEND:
                    modality = event.get("modality", "")
                    if modality not in self._capabilities.input_modalities:
                        await emit(ev.error(f"unsupported input modality: {modality}"))
                        continue
                    await self.adapter.on_input(self.session, modality, event.get("data"))
                    self.session.state = DuplexState.LISTENING
                    task = self._ensure_started(task, emit)
                elif etype in (ev.INPUT_COMMIT, ev.RESPONSE_CREATE):
                    task = self._ensure_started(task, emit)
                elif etype == ev.RESPONSE_CANCEL:
                    # Barge-in is native for lockstep models: bump the epoch for the
                    # playback cursor but keep the single response streaming.
                    await self._barge_in()
                    await emit(ev.cancelled(self.session.response_index))
                elif etype == ev.PLAYBACK_ACK:
                    await self.adapter.on_playback_ack(self.session, int(event.get("cursor", 0)))
                elif etype == ev.CLOSE:
                    break
        finally:
            # Drain on EVERY exit path: CLOSE, iterator EOF, transport errors,
            # and cancellation must not leave the eternal response blocked on
            # its inbox or the session outside CLOSED.
            try:
                if task is not None and not task.done():
                    await self.adapter.on_close(self.session)
                    await task
            finally:
                self.session.state = DuplexState.CLOSED

    def _ensure_started(self, task: asyncio.Task | None, emit: Emit) -> asyncio.Task:
        # Start the single eternal response exactly once; never supersede it.
        if task is None or task.done():
            return asyncio.create_task(self._respond_once(emit))
        return task

    async def _respond_once(self, emit: Emit) -> None:
        # One response for the whole session. No staleness checks: the loop ends
        # only when the adapter's respond() returns (after on_close drains its
        # input).
        response_index, _ = self.session.begin_response()
        await emit(ev.created(response_index))
        try:
            async for chunk in self.adapter.respond(self.session):
                await emit(ev.delta(response_index, chunk.modality, chunk.data))
        except asyncio.CancelledError:
            raise
        except Exception as err:
            await emit(ev.error(f"response failed: {err}"))
        finally:
            await emit(ev.done(response_index))
            self.session.end_response()
