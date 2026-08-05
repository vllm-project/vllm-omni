# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-free tests for the batched PersonaPlex slot manager (stub engine)."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.personaplex.engine import FrameOutput, PrefillStep
from vllm_omni.experimental.fullduplex.personaplex.serving.batched import (
    BatchedSessionManager,
    SlotPhase,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAME = 4


class StubEngine:
    """Records the slot-API call sequence; echoes each row's pcm as its output."""

    frame_size = FRAME

    def __init__(self, batch_size: int = 4, prefill_len: int = 2) -> None:
        self.config = SimpleNamespace(batch_size=batch_size)
        self.prefill_len = prefill_len
        self.recycled: list[int] = []
        self.voice_ended: list[int] = []
        self.prefill_done: list[int] = []
        self.steps: list[tuple[np.ndarray, dict | None]] = []

    def prefill_steps(self, persona=None):
        steps = [PrefillStep("sacrifice"), PrefillStep("voice", embedding=None), PrefillStep("voice_end")]
        steps += [PrefillStep("tokens", moshi_tokens=None, text_token=3, user_sine=True)] * self.prefill_len
        return steps

    def recycle_slot(self, b):
        self.recycled.append(b)

    def voice_end(self, b):
        self.voice_ended.append(b)

    def end_prefill(self, b):
        self.prefill_done.append(b)

    def step_batch(self, pcm, prefill=None):
        self.steps.append((pcm.copy(), dict(prefill) if prefill else None))
        return [FrameOutput(audio=pcm[b].copy(), text=None) for b in range(self.config.batch_size)]


def test_all_idle_does_not_step():
    eng = StubEngine()
    mgr = BatchedSessionManager(eng)
    assert mgr.tick() is False
    assert eng.steps == []


def test_pristine_slot_is_live_immediately():
    eng = StubEngine()
    mgr = BatchedSessionManager(eng)
    slot, ready_now = mgr.acquire()
    assert ready_now is True
    assert slot.phase is SlotPhase.LIVE
    assert eng.recycled == []


def test_custom_persona_prefills_and_recycles_on_tick_thread():
    eng = StubEngine(prefill_len=2)
    mgr = BatchedSessionManager(eng)
    slot, ready_now = mgr.acquire(persona="custom")
    assert ready_now is False
    assert slot.phase is SlotPhase.PREFILL
    assert eng.recycled == []  # deferred: reset must run between engine steps

    ready = []
    slot.on_ready = lambda: ready.append(True)

    mgr.tick()  # recycle + sacrifice
    assert eng.recycled == [slot.index]
    assert eng.steps[-1][1][slot.index].kind == "sacrifice"

    mgr.tick()  # voice frame
    assert eng.steps[-1][1][slot.index].kind == "voice"
    assert eng.voice_ended == []

    mgr.tick()  # voice_end executed BETWEEN ticks, then first tokens frame
    assert eng.voice_ended == [slot.index]
    assert eng.steps[-1][1][slot.index].kind == "tokens"

    mgr.tick()  # second tokens frame
    assert not ready
    mgr.tick()  # prefill exhausted -> live
    assert eng.prefill_done == [slot.index]
    assert slot.phase is SlotPhase.LIVE
    assert ready == [True]


def test_pristine_slot_after_ticking_requires_recycle():
    # step_batch advances EVERY row each tick, so a pristine idle row that has
    # lived through ticks has drifted from its boot prefill and must be
    # recycled before a client may join it.
    eng = StubEngine()
    mgr = BatchedSessionManager(eng)
    first, ready_now = mgr.acquire()
    assert ready_now is True  # claimed before any tick: truly pristine
    mgr.tick()
    second, ready_now2 = mgr.acquire()
    assert second.index != first.index
    assert ready_now2 is False
    assert second.phase is SlotPhase.PREFILL


def test_used_slot_requires_recycle_even_with_default_persona():
    eng = StubEngine()
    mgr = BatchedSessionManager(eng)
    slot, ready_now = mgr.acquire()
    assert ready_now
    mgr.release(slot)
    slot2, ready_now2 = mgr.acquire()
    assert slot2.index == slot.index
    assert ready_now2 is False
    assert slot2.phase is SlotPhase.PREFILL


def test_acquire_returns_none_when_full():
    eng = StubEngine(batch_size=2)
    mgr = BatchedSessionManager(eng)
    assert mgr.acquire() is not None
    assert mgr.acquire() is not None
    assert mgr.acquire() is None


def test_feed_frames_epoch_guard_and_backlog_cap():
    eng = StubEngine()
    mgr = BatchedSessionManager(eng, max_buffered_frames=3)
    slot, _ = mgr.acquire()
    mgr.feed(slot, np.arange(FRAME * 2, dtype=np.float32), slot.epoch)
    assert len(slot.inq) == 2
    mgr.feed(slot, np.zeros(3, dtype=np.float32), slot.epoch)  # partial tail buffered
    assert len(slot.inq) == 2 and slot.pending.shape[0] == 3
    mgr.feed(slot, np.zeros(FRAME * 5, dtype=np.float32), slot.epoch)
    assert len(slot.inq) == 3  # oldest dropped beyond the cap
    mgr.feed(slot, np.zeros(FRAME, dtype=np.float32), slot.epoch - 1)  # stale epoch ignored
    assert len(slot.inq) == 3


def test_live_slot_consumes_input_and_emits_output():
    eng = StubEngine()
    mgr = BatchedSessionManager(eng)
    slot, _ = mgr.acquire()
    got = []
    slot.on_output = got.append
    frame = np.full(FRAME, 0.5, dtype=np.float32)
    mgr.feed(slot, frame, slot.epoch)
    mgr.tick()
    assert len(got) == 1
    np.testing.assert_array_equal(got[0].audio, frame)
    # queue drained: next tick feeds zeros
    mgr.tick()
    np.testing.assert_array_equal(got[1].audio, np.zeros(FRAME, dtype=np.float32))


def test_release_mid_flight_stops_emission():
    eng = StubEngine()
    mgr = BatchedSessionManager(eng)
    slot, _ = mgr.acquire()
    got = []
    slot.on_output = got.append
    mgr.tick()
    assert len(got) == 1
    mgr.release(slot)
    # another slot keeps the batch non-idle
    other, _ = mgr.acquire()  # same slot index, now PREFILL (used)
    assert other.phase is SlotPhase.PREFILL
    mgr.tick()
    assert len(got) == 1  # no further output for the released epoch
