# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Paced onset (``duplex_session.pacing``) through the production continuation scheduler.

Same fake clock as ``test_deadline_pacing_scheduler``: ``asyncio.sleep``
advances it, so the submission time of each continuation is exact.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import vllm_omni.engine.duplex.session.runner as runner_module
from tests.engine.duplex.test_deadline_pacing_scheduler import (
    FakeClock,
    _active_response_harness,
    _continuation_kwargs,
    _install_fake_clock,
)
from tests.engine.duplex.test_session_runner import (
    LISTEN_TOKEN_ID,
    append_audio,
    close_harness,
    find,
    listen_output,
    open_harness,
    tts_output,
    types,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.session.pacing import PaceState, SessionPacing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _paced(h, **pacing: object) -> SessionPacing:
    """Install a pacing state on an open harness (the runtime config of ``open_harness`` has none)."""
    config: dict[str, object] = {"enabled": True, "onset_lead_max_s": 0.9}
    config.update(pacing)
    pace = SessionPacing(DuplexSessionRuntimeConfig(pacing=config), h.session)
    h.runner.run.pace = pace
    return pace


async def _continue(h, **kwargs: object) -> None:
    scheduled = await h.runner._schedule_silence_continuation(
        h.runner.model.silence_unit_payload(),
        **_continuation_kwargs(h),
        **kwargs,
    )
    assert scheduled is True
    task = h.runner.tasks.append_tail
    assert task is not None
    assert await task


@pytest.mark.asyncio
async def test_pacing_is_off_by_default() -> None:
    h = await open_harness()
    try:
        assert h.runner.run.pace is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_runner_creates_pacing_when_enabled() -> None:
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(pacing={"enabled": True}))
    try:
        assert isinstance(h.runner.run.pace, SessionPacing)
        assert h.runner.run.pace.state == PaceState.LISTEN
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_quiet_client_submits_ahead_of_the_nominal_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(start=100.0)
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=asyncio.sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        pace = _paced(h)
        pace.on_commit(100.0)  # the harness set-up advanced the fake clock
        assert pace.state == PaceState.ONSET
        # Unit 1 submitted at 100.0; its segment ends at 100.2, inside the
        # 0.25 s quiet guard: unit 2 goes once the guard passes, not at 101.0,
        # and the nominal chain is untouched (next deadline 102.0).
        state.last_native_submit_monotonic = 100.0
        state.silence_deadline_monotonic = None
        clock.value = 100.2
        await _continue(h)
        assert state.last_native_submit_monotonic == pytest.approx(100.25)
        assert state.silence_deadline_monotonic == pytest.approx(102.0)

        # The first chunk was 0.84 s: lead 0.26, unit 3 goes at 102 - 0.26.
        pace.on_audio_emit(100.5, 0.84)
        clock.value = 100.6
        await _continue(h)
        assert state.last_native_submit_monotonic == pytest.approx(101.74)
        assert state.silence_deadline_monotonic == pytest.approx(103.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_lead_rechecked_while_waiting_only_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """d1 arriving mid-wait shrinks the lead: the unit fires later, never earlier."""
    clock = FakeClock(start=200.0)
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=asyncio.sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        pace = _paced(h)
        pace.on_commit(200.0)
        state.last_native_submit_monotonic = 200.0
        state.silence_deadline_monotonic = 202.0
        clock.value = 200.5
        real_speech_fire_at = pace.speech_fire_at
        calls = {"n": 0}

        def speech_fire_at(nominal: float, now: float) -> float:
            calls["n"] += 1
            if calls["n"] == 3:
                # d1 lands while the unit waits for 202.0 - 0.9.
                pace.on_audio_emit(now, 1.0)
            return real_speech_fire_at(nominal, now)

        monkeypatch.setattr(pace, "speech_fire_at", speech_fire_at)
        await _continue(h)
        assert state.last_native_submit_monotonic == pytest.approx(201.9)
        assert state.silence_deadline_monotonic == pytest.approx(203.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_an_append_inside_the_quiet_guard_keeps_unit_2_on_the_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mic-open client (200 ms chunks) whose next append lands after unit 1's segment ended."""
    clock = FakeClock(start=99.0)
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=asyncio.sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        pace = _paced(h)
        for t in (99.4, 99.6, 99.8):
            pace.on_client_append(t, is_speech=True, duration_s=0.2)
        pace.on_commit(100.0)  # guard 1.5 x 0.2 = 0.3 s
        state.last_native_submit_monotonic = 100.0
        state.silence_deadline_monotonic = None
        clock.value = 100.15
        real_speech_fire_at = pace.speech_fire_at

        def speech_fire_at(nominal: float, now: float) -> float:
            if now >= 100.2 and pace.quiet_since_commit:
                pace.on_client_append(now, is_speech=False, duration_s=0.2)
            return real_speech_fire_at(nominal, now)

        monkeypatch.setattr(pace, "speech_fire_at", speech_fire_at)
        await _continue(h)
        assert pace.quiet_since_commit is False
        assert state.last_native_submit_monotonic == pytest.approx(101.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_input_after_commit_restores_the_wall_clock_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(start=300.0)
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=asyncio.sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        pace = _paced(h)
        pace.on_commit(clock.value)
        pace.on_client_append(clock.value, is_speech=False, duration_s=0.2)
        assert pace.lead_s(clock.value + 1.0) == 0.0
        state.last_native_submit_monotonic = 300.0
        state.silence_deadline_monotonic = None
        clock.value = 300.2
        await _continue(h)
        assert state.last_native_submit_monotonic == pytest.approx(301.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_runner_reports_client_appends_to_the_pacing_state() -> None:
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(pacing={"enabled": True}))
    try:
        pace = h.runner.run.pace
        assert isinstance(pace, SessionPacing)
        pace.on_commit(0.0)
        assert pace.quiet_since_commit is True
        await h.run(append_audio(3200, value=0.0, is_speech=False))
        assert pace.quiet_since_commit is False
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_idle_continuations_keep_the_nominal_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(start=400.0)
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=asyncio.sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        pace = _paced(h)
        pace.on_commit(clock.value)
        state.last_native_submit_monotonic = 400.0
        state.silence_deadline_monotonic = None
        clock.value = 400.2
        await _continue(h, pace_kind="idle")
        assert state.last_native_submit_monotonic == pytest.approx(401.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_same_turn_response_does_not_stale_a_paced_model_turn_unit() -> None:
    h = await _active_response_harness()
    try:
        session = h.session
        # The response that opened while unit 2 waited is the unit's own turn's.
        session.bind_response_turn(session.turn_id)
        turn_id = session.turn_id
        kwargs = {
            "request_id": session.active_request_id,
            "response_id": None,
            "response_owned": False,
            "expected_epoch": session.epoch,
            "expected_model_turn_id": turn_id,
        }
        # Historical rule: any active response makes a model-turn continuation stale.
        assert h.runner.model.silence_continuation_is_stale(**kwargs) is True
        _paced(h)  # skip switch off: unchanged
        assert h.runner.model.silence_continuation_is_stale(**kwargs) is True
        _paced(h, onset_skip_response_wait=True)
        assert h.runner.model.silence_continuation_is_stale(**kwargs) is False
        # A response of another turn still makes it stale.
        assert h.runner.model.silence_continuation_is_stale(**{**kwargs, "expected_model_turn_id": turn_id + 1}) is True
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stop_token_trigger_skips_the_response_wait_on_a_paced_onset(monkeypatch: pytest.MonkeyPatch) -> None:
    h = await open_harness()
    try:
        session = h.session
        session.bind_request(h.stage0_request_id())
        assert session.active_response_id is None
        calls: list[dict] = []

        async def fake_continue(**kwargs) -> None:
            calls.append(kwargs)

        monkeypatch.setattr(h.runner.model, "maybe_continue_response", fake_continue)
        sleeps: list[float] = []
        real_sleep = asyncio.sleep

        async def counting_sleep(delay: float) -> None:
            sleeps.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(runner_module.asyncio, "sleep", counting_sleep)
        pace = _paced(h, onset_skip_response_wait=True)
        pace.on_commit(time.monotonic() - 1.0)  # past the quiet guard
        await h.runner._continue_response_on_stop_token_task(session.epoch)
        assert sleeps == []
        assert calls == [{"expected_epoch": session.epoch, "expected_model_turn_id": session.turn_id}]
    finally:
        monkeypatch.undo()
        await close_harness(h)


@pytest.mark.asyncio
async def test_paced_unit_snaps_to_the_fire_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(start=100.0)
    _install_fake_clock(monkeypatch, clock=clock, real_sleep=asyncio.sleep)
    h = await _active_response_harness()
    state = h.runner.model_state
    try:
        pace = _paced(h, fire_grid_ms=250)
        pace.on_commit(100.0)
        pace.on_audio_emit(100.5, 0.84)
        assert pace.state == PaceState.PACED
        state.last_native_submit_monotonic = 101.0
        state.silence_deadline_monotonic = 102.0
        clock.value = 100.6
        await _continue(h)
        # Lead 0.26 gives 101.74; the grid moves it to 101.5 (lead 0.5 <= 0.9).
        assert state.last_native_submit_monotonic == pytest.approx(101.5)
        assert state.silence_deadline_monotonic == pytest.approx(103.0)
    finally:
        await close_harness(h)


async def _speaking_quiet_client(pacing: dict[str, object]):
    """A committed client 1 whose reply is speaking (TTS in flight), with ``abort_on_model_listen``."""
    h = await open_harness(
        runtime_config=DuplexSessionRuntimeConfig(abort_on_model_listen=True, pacing=pacing),
    )
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
    assert h.session.active_response_id is not None
    return h, request_id


async def _model_listens(h, request_id: str):
    return await h.deliver_and_settle(
        listen_output(request_id),
        stage_id=0,
        segment_finished=True,
        segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
        segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
    )


@pytest.mark.asyncio
async def test_a_quiet_clients_model_listen_does_not_abort_the_paced_reply() -> None:
    """Unit 2 decides listen while unit 1 is still synthesized: no barge-in, the reply drains."""
    h, request_id = await _speaking_quiet_client({"enabled": True, "onset_lead_max_s": 0.9})
    try:
        h.runner.run.pace.on_commit(time.monotonic())
        response_id = h.session.active_response_id
        events = await _model_listens(h, request_id)
        assert h.port.aborts == []
        assert h.session.epoch == 0
        assert "audio.cancelled" not in types(events)
        assert all(event.status != "cancelled" for event in events if event.type == "response.done")
        assert h.session.active_response_id == response_id
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_streaming_clients_model_listen_still_aborts() -> None:
    h, request_id = await _speaking_quiet_client({"enabled": True, "onset_lead_max_s": 0.9})
    try:
        pace = h.runner.run.pace
        pace.on_commit(time.monotonic())
        pace.on_client_append(time.monotonic(), is_speech=True, duration_s=0.2)  # input after the commit
        events = await _model_listens(h, request_id)
        assert find(events, "response.done").status == "cancelled"
        assert h.session.epoch == 1
    finally:
        await close_harness(h)
