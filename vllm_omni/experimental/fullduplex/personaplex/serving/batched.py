# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot manager + lockstep tick loop for batched PersonaPlex serving.

Pure-python orchestration over a :class:`PersonaPlexEngine`-shaped object
(injected, so tests run GPU-free with a stub): a fixed table of B conversation
slots, one shared wall-clock 80 ms tick that advances every non-idle slot, and
per-slot recycle prefill for new callers.

Threading model: ``tick()`` runs on a dedicated stepping thread (the engine call
is synchronous CUDA work); ``acquire``/``release``/``feed`` are called from the
asyncio thread. All slot state is guarded by one lock; per-frame output is
handed off via each slot's ``on_output`` callback (called on the stepping
thread -- the server bridges to asyncio with ``call_soon_threadsafe``).
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SlotPhase(enum.Enum):
    IDLE = "idle"
    PREFILL = "prefill"
    LIVE = "live"


@dataclass
class Slot:
    index: int
    phase: SlotPhase = SlotPhase.IDLE
    used: bool = False  # a used slot must be recycled before reuse
    needs_recycle: bool = False  # engine.reset executed by the tick thread only
    epoch: int = 0
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    inq: deque = field(default_factory=deque)
    prefill: deque = field(default_factory=deque)
    on_output: Callable[[Any], None] | None = None  # FrameOutput, stepping thread
    on_ready: Callable[[], None] | None = None  # prefill finished, stepping thread


class BatchedSessionManager:
    """Own the slot table and drive one engine tick per call to :meth:`tick`."""

    def __init__(self, engine, max_buffered_frames: int = 25) -> None:
        self.engine = engine
        self.frame_size = engine.frame_size
        self.batch_size = engine.config.batch_size
        self.max_buffered_frames = max_buffered_frames
        self.slots = [Slot(index=i) for i in range(self.batch_size)]
        self.lock = threading.Lock()
        # step_batch advances EVERY row each tick (lockstep), so once any tick has
        # run, "pristine" idle rows have silently drifted from their boot prefill
        # and must be recycled before a client may join them.
        self._ticked = False

    # -- connection side (asyncio thread) ------------------------------------

    def acquire(self, persona: str | None = None) -> tuple[Slot, bool] | None:
        """Claim an idle slot. Returns ``(slot, ready_now)`` or ``None`` if full.

        A pristine slot (never used since boot, and claimed before the first
        engine tick) still carries the default voice + persona from
        ``open_batch`` and is live immediately when no custom persona is
        requested. Otherwise the slot is recycled and its system prompt
        replayed over the next ticks (``ready_now == False``) -- including
        pristine rows after ticking has started, because lockstep stepping
        advances every row whether or not a client is attached.
        """
        with self.lock:
            slot = next((s for s in self.slots if s.phase is SlotPhase.IDLE), None)
            if slot is None:
                return None
            slot.epoch += 1
            slot.pending = np.zeros(0, dtype=np.float32)
            slot.inq.clear()
            slot.prefill.clear()
            if not slot.used and persona is None and not self._ticked:
                slot.phase = SlotPhase.LIVE
                slot.used = True
                return slot, True
            # The engine reset itself must run BETWEEN engine steps, i.e. on the
            # stepping thread -- only mark it here; tick() executes it.
            slot.prefill.extend(self.engine.prefill_steps(persona))
            slot.needs_recycle = True
            slot.phase = SlotPhase.PREFILL
            slot.used = True
            return slot, False

    def release(self, slot: Slot) -> None:
        with self.lock:
            slot.phase = SlotPhase.IDLE
            slot.epoch += 1
            slot.on_output = None
            slot.on_ready = None
            slot.inq.clear()
            slot.prefill.clear()
            slot.pending = np.zeros(0, dtype=np.float32)

    def feed(self, slot: Slot, pcm: np.ndarray, epoch: int) -> None:
        """Buffer incoming PCM and carve it into exact engine frames."""
        with self.lock:
            if slot.epoch != epoch or slot.phase is SlotPhase.IDLE:
                return
            samples = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
            slot.pending = np.concatenate([slot.pending, samples]) if slot.pending.size else samples
            n = self.frame_size
            while slot.pending.shape[0] >= n:
                frame, slot.pending = slot.pending[:n], slot.pending[n:]
                slot.inq.append(frame)
                while len(slot.inq) > self.max_buffered_frames:
                    slot.inq.popleft()  # realtime: drop the oldest backlog

    # -- stepping side (dedicated thread) ------------------------------------

    def tick(self) -> bool:
        """Advance every non-idle slot one frame. Returns False when all idle."""
        with self.lock:
            if all(s.phase is SlotPhase.IDLE for s in self.slots):
                return False
            self._ticked = True
            pcm = np.zeros((self.batch_size, self.frame_size), dtype=np.float32)
            prefill: dict[int, Any] = {}
            emitters: list[tuple[Slot, int]] = []
            for slot in self.slots:
                if slot.phase is SlotPhase.LIVE:
                    if slot.inq:
                        pcm[slot.index] = slot.inq.popleft()
                    emitters.append((slot, slot.epoch))
                elif slot.phase is SlotPhase.PREFILL:
                    if slot.needs_recycle:
                        self.engine.recycle_slot(slot.index)
                        slot.needs_recycle = False
                    while slot.prefill and slot.prefill[0].kind == "voice_end":
                        self.engine.voice_end(slot.index)
                        slot.prefill.popleft()
                    if slot.prefill:
                        prefill[slot.index] = slot.prefill.popleft()
                    else:
                        self.engine.end_prefill(slot.index)
                        slot.phase = SlotPhase.LIVE
                        emitters.append((slot, slot.epoch))
                        if slot.on_ready is not None:
                            slot.on_ready()

        outs = self.engine.step_batch(pcm, prefill or None)

        with self.lock:
            for slot, epoch in emitters:
                out = outs[slot.index]
                if out is None or slot.epoch != epoch or slot.on_output is None:
                    continue
                slot.on_output(out)
        return True

    def run_forever(self, stop: threading.Event, frame_seconds: float = 0.08) -> None:
        """Wall-clock-paced stepping loop (run on a dedicated thread)."""
        next_t = time.monotonic()
        while not stop.is_set():
            did = self.tick()
            if not did:
                time.sleep(frame_seconds / 4)
                next_t = time.monotonic()
                continue
            next_t += frame_seconds
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:
                # Fell far behind realtime (e.g. B too large for the GPU); resync
                # rather than bursting to catch up.
                logger.warning("batched tick loop %.1fs behind realtime; resyncing", -delay)
                next_t = time.monotonic()
