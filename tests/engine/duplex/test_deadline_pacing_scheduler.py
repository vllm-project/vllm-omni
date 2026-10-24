# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Focused scheduler tests for deadline-aligned silence continuation.

These tests exercise the production ``_schedule_silence_continuation`` path
(and the append acceptance callback it installs) rather than copying the
scheduling arithmetic. The monotonic clock is a fake so the deadline math is
deterministic, and ``asyncio.sleep`` is simulated by advancing that clock.
Append completion is controlled by gating the stage port's ``submit``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

import vllm_omni.engine.duplex.session.model_channel as model_channel_module
import vllm_omni.engine.duplex.session.runner as runner_module
from tests.engine.duplex.test_session_runner import (
    Harness,
    append_audio,
    close_harness,
    open_harness,
    tts_output,
)
from vllm_omni.engine.duplex.contracts import DuplexStageSubmission

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK_PERIOD_S = 1.0


class FakeClock:
    """A monotonic clock whose value the test controls directly."""

    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value


async def _active_response_harness() -> Harness:
    """Open a harness and drive a TTS segment so a response is active."""
    h = await open_harness()
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
    assert h.session.active_response_id is not None
    return h


def _continuation_kwargs(h: Harness) -> dict[str, object]:
    """The scheduler arguments the model channel would pass for this session."""
    return {
        "request_id": h.session.active_request_id,
        "owner_id": f"response:{h.session.active_response_id}",
        "response_id": h.session.active_response_id,
        "response_owned": True,
        "expected_epoch": h.session.epoch,
        "expected_model_turn_id": h.session.turn_id,
    }


def _install_fake_clock(
    monkeypatch: pytest.MonkeyPatch,
    *,
    clock: FakeClock,
    real_sleep: Callable[..., Any],
) -> list[float]:
    """Point both modules at the fake clock and simulate ``asyncio.sleep``.

    ``real_sleep`` is the unpatched ``asyncio.sleep`` captured before patching;
    the simulated sleep advances the fake clock by the requested delay and then
    yields control so the append task can run. The returned ``stall`` list is
    added once to the next *positive* sleep so a test can model a wake that
    happens late (after the planned deadline); zero-duration scheduling yields
    do not consume it.
    """
    monkeypatch.setattr(runner_module.time, "monotonic", clock)
    monkeypatch.setattr(model_channel_module.time, "monotonic", clock)
    stall: list[float] = [0.0]

    async def fake_sleep(delay: float) -> None:
        clock.value += delay
        if delay > 0:
            clock.value += stall[0]
            stall[0] = 0.0
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return stall


@pytest.mark.asyncio
async def test_normal_cadence_keeps_deadlines_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Processing time reduces the remaining sleep; consecutive deadlines align.

    Each accepted unit becomes the new anchor (its submission time), and the
    following continuation advances from the stored deadline rather than
    resetting to now.
    """
    clock = FakeClock(start=100.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    try:
        # Real input accepted at t=100.0; the model processes for 0.4 s.
        h.runner.model_state.last_native_submit_monotonic = 100.0
        clock.value = 100.4

        # First continuation includes 5 ms of input-arrival grace.
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        first_task = h.runner.tasks.append_tail
        assert first_task is not None
        assert await first_task
        assert h.runner.model_state.last_native_submit_monotonic == pytest.approx(101.005)
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(102.005)

        # Model produces the continuation's audio; the next unit is planned
        # from the stored deadline; the initial grace does not accumulate.
        clock.value = 101.3
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        second_task = h.runner.tasks.append_tail
        assert second_task is not None
        assert await second_task
        assert h.runner.model_state.silence_deadline_monotonic == pytest.approx(103.005)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_in_flight_real_append_skips_outdated_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real append accepted after the silence was planned re-anchors the chain.

    The planned silence queues behind the real append; once the real append
    completes and re-anchors, ``before_append`` detects the changed anchor
    (numeric timestamp) and the outdated silence is skipped (no submission,
    no callback overwrite).
    """
    clock = FakeClock(start=200.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    submissions_before = len(h.port.submissions)

    original_submit = h.port.submit
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_submit(submission: DuplexStageSubmission) -> Any:
        entered.set()
        await release.wait()
        return await original_submit(submission)

    monkeypatch.setattr(h.port, "submit", gated_submit)
    try:
        # Existing anchor at t=200.0.
        state.last_native_submit_monotonic = 200.0
        state.silence_deadline_monotonic = None

        # The real append starts at t+0.8 and blocks inside the gate.
        clock.value = 200.8
        real_append_task = await h.runner._start_append(
            h.runner.model.silence_unit_payload(),
            final=False,
            silence_continuation=False,
        )
        await entered.wait()
        assert not real_append_task.done()

        # At t+1.0 the silence is planned using the existing anchor; its append
        # task waits behind the real append.
        clock.value = 201.0
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        silence_task = h.runner.tasks.append_tail
        assert silence_task is not None
        assert not silence_task.done()

        # At t+1.1 the real append is released; both appends finish.
        clock.value = 201.1
        release.set()
        assert await real_append_task
        assert await silence_task

        # Exactly one additional submission happened: the real append. The
        # outdated silence was skipped, so its callback never overwrote the
        # new anchor (the real append's submission time t+0.8), and the
        # deadline the real append cleared stays cleared.
        assert len(h.port.submissions) == submissions_before + 1
        assert state.last_native_submit_monotonic == pytest.approx(200.8)
        assert state.silence_deadline_monotonic is None

        # A fresh continuation uses the new anchor. The clock is left at t+1.1:
        # A fresh input-arrival grace makes the next deadline t+1.805.
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        fresh_task = h.runner.tasks.append_tail
        assert fresh_task is not None
        assert await fresh_task
        assert state.last_native_submit_monotonic == pytest.approx(201.805)
        assert state.silence_deadline_monotonic == pytest.approx(202.805)
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_long_stall_saves_future_deadline_from_actual_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submission more than one period past the planned deadline restarts.

    The schedule plans the following deadline at ``planned + period``, but the
    actual submission happens much later. The accepted continuation must save
    ``submit_time + period`` so the next unit is not immediately due.
    """
    clock = FakeClock(start=300.0)
    real_sleep = asyncio.sleep
    stall = _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        # Plan a continuation whose deadline is 1.0 s in the future. The wake
        # is delayed 2 s past the planned submission (anchor + 3.0 during the
        # scheduler's sleep), so the append accepts 2 s late.
        anchor = clock.value
        state.last_native_submit_monotonic = anchor
        state.silence_deadline_monotonic = None
        stall[0] = 2.0
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        append_task = h.runner.tasks.append_tail
        assert append_task is not None
        assert await append_task
        # A long stall restarts from the actual submission, including its grace.
        assert state.last_native_submit_monotonic == pytest.approx(anchor + 3.005)
        assert state.silence_deadline_monotonic == pytest.approx(anchor + 4.005)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_small_wakeup_delay_preserves_planned_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary wakeup jitter (within one period) keeps the planned cadence.

    A submission 0.2 s past the planned deadline must not reset the chain: the
    stored deadline stays at ``planned + period`` so drift does not accumulate.
    """
    clock = FakeClock(start=400.0)
    real_sleep = asyncio.sleep
    stall = _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        anchor = clock.value
        state.last_native_submit_monotonic = anchor
        state.silence_deadline_monotonic = None
        stall[0] = 0.2

        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        append_task = h.runner.tasks.append_tail
        assert append_task is not None
        assert await append_task

        # Submitted at anchor + period + 0.2 (within one period): the planned
        # following deadline is kept.
        assert state.last_native_submit_monotonic == pytest.approx(anchor + CHUNK_PERIOD_S + 0.205)
        assert state.silence_deadline_monotonic == pytest.approx(anchor + 2 * CHUNK_PERIOD_S + 0.005)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_exact_one_period_late_preserves_planned_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly one period late is not "more than one period late".

    The strict ``>`` comparison means a submission exactly at the planned
    following deadline keeps the planned cadence (no extra catch-up unit).
    """
    clock = FakeClock(start=500.0)
    real_sleep = asyncio.sleep
    stall = _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        anchor = clock.value
        state.last_native_submit_monotonic = anchor
        state.silence_deadline_monotonic = None
        stall[0] = 1.0

        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        append_task = h.runner.tasks.append_tail
        assert append_task is not None
        assert await append_task

        # Submitted exactly at the planned following deadline: kept.
        assert state.last_native_submit_monotonic == pytest.approx(anchor + 2 * CHUNK_PERIOD_S + 0.005)
        assert state.silence_deadline_monotonic == pytest.approx(anchor + 2 * CHUNK_PERIOD_S + 0.005)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_failed_append_leaves_timing_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed append never advances the timing chain."""
    clock = FakeClock(start=600.0)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        state.last_native_submit_monotonic = 600.0
        state.silence_deadline_monotonic = 601.0
        before_anchor = state.last_native_submit_monotonic
        before_deadline = state.silence_deadline_monotonic
        h.port.fail_submit = RuntimeError("boom")

        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(),
            **_continuation_kwargs(h),
        )
        assert scheduled is True
        append_task = h.runner.tasks.append_tail
        assert append_task is not None
        # The append fails: its task reports failure, and timing is untouched.
        assert await append_task is False
        assert state.last_native_submit_monotonic == before_anchor
        assert state.silence_deadline_monotonic == before_deadline
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_queued_audio_blocks_silence_before_pcm_buffering() -> None:
    h = await open_harness()
    try:
        h.submit(append_audio())
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert not h.runner.model_state.audio_buffer.has_reserved()
        assert h.runner._real_input_waiting()
        await h.settle()
        assert not h.runner._real_input_waiting()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_audio", [False, True])
async def test_decoding_audio_blocks_silence_until_handled(monkeypatch, invalid_audio) -> None:
    h = await open_harness()
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.runner.offload

    async def convert(*args, **kwargs):
        entered.set()
        await release.wait()
        if invalid_audio:
            raise ValueError("invalid audio")
        return await original(*args, **kwargs)

    monkeypatch.setattr(h.runner, "offload", convert)
    try:
        h.submit(append_audio())
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert h.session.pending_input_bytes == 0
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert not h.runner.model_state.audio_buffer.has_reserved()
        assert h.runner._real_input_waiting()
        release.set()
        await h.settle()
        assert not h.runner._real_input_waiting()
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_input_boundary_grace_does_not_submit_silence(monkeypatch) -> None:
    clock = FakeClock(start=700)
    real_sleep = asyncio.sleep
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=real_sleep)
    h = await _active_response_harness()
    anchor = clock.value
    h.runner.model_state.last_native_submit_monotonic = anchor
    h.runner.model_state.silence_deadline_monotonic = None
    nominal_deadline = anchor + CHUNK_PERIOD_S
    clock.value = nominal_deadline
    arrival = nominal_deadline + 0.002
    received = False

    async def sleep_with_input(delay):
        nonlocal received
        wake = clock.value + delay
        if not received and clock.value < arrival <= wake:
            clock.value = arrival
            h.submit(append_audio())
            received = True
            await real_sleep(0)
        clock.value = wake
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep_with_input)
    try:
        scheduled = await h.runner._schedule_silence_continuation(
            h.runner.model.silence_unit_payload(), **_continuation_kwargs(h)
        )
        assert scheduled is False
        assert received
        assert h.runner.model_state.silence_deadline_monotonic is None
    finally:
        await close_harness(h)
