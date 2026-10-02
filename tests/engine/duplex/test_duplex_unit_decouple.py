# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Duplex unit continuation decoupling (VLLM_OMNI_DUPLEX_CONTINUE_ON / _UNIT_DEPTH).

The historical chain plans a spoken unit's next append only after the unit's
audio deltas have been emitted, which holds each session to one unit in
flight. These tests pin the opt-in behaviours and the invariants that must
not change when the trigger moves earlier:

* stop_token mode schedules the next unit's append while the current unit's
  audio has not even been produced yet, without double-scheduling when the
  audio-side trigger later fires for the same unit;
* client audio emission order is untouched (it follows stage-output arrival,
  not append submission time) even when a later unit's append is submitted
  before an earlier unit's audio is emitted;
* default (audio / depth 1) behaviour matches the historical trigger exactly;
* depth >= 2 lets a second continuation be planned while the first is still
  submitting, where depth 1 keeps waiting.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from tests.engine.duplex.test_session_runner import (
    LISTEN_TOKEN_ID,
    Harness,
    append_audio,
    close_harness,
    open_harness,
    tts_output,
    types,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CONTINUE_ON_ENV = "VLLM_OMNI_DUPLEX_CONTINUE_ON"
DEPTH_ENV = "VLLM_OMNI_DUPLEX_UNIT_DEPTH"


def spoken_stage0_output(request_id: str) -> SimpleNamespace:
    """A finished spoken Stage-0 segment: stopped on the chunk terminator.

    The last token is not the listen token, so the plugin's ``decide_output``
    returns None (forwarded to the TTS stage) -- exactly the output whose stop
    token is the decoupled continuation trigger.
    """
    return SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[SimpleNamespace(text="", token_ids=[21, 22, 23], stop_reason=23, multimodal_output={})],
        multimodal_output={"meta.listen_token_id": LISTEN_TOKEN_ID},
    )


async def _active_response_harness() -> Harness:
    """Open a harness and drive one TTS segment so a response is active."""
    h = await open_harness()
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
    assert h.session.active_response_id is not None
    assert len(h.port.submissions) == 1
    return h


def _make_continuation_due_now(h: Harness) -> None:
    """Point the silence-continuation anchor far enough back that no sleep is owed."""
    model_state = h.runner.model_state
    model_state.last_native_submit_monotonic = time.monotonic() - 1.2
    model_state.silence_deadline_monotonic = None


def _continuation_kwargs(h: Harness) -> dict[str, Any]:
    return {
        "request_id": h.session.active_request_id,
        "owner_id": f"response:{h.session.active_response_id}",
        "response_id": h.session.active_response_id,
        "response_owned": True,
        "expected_epoch": h.session.epoch,
        "expected_model_turn_id": h.session.turn_id,
    }


# --------------------------------------------------------------------------- #
# Trigger switch                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stop_token_mode_schedules_continuation_before_audio_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Stage-0 segment finish alone must plan the next unit's append.

    At the moment the assertion runs, the spoken unit's audio has not been
    produced at all (no Stage-1 output was delivered), yet the next silence
    unit is already submitted -- the decoupling this switch exists for.
    """
    monkeypatch.setenv(CONTINUE_ON_ENV, "stop_token")
    h = await _active_response_harness()
    try:
        request_id = h.stage0_request_id()
        _make_continuation_due_now(h)

        h.deliver(
            spoken_stage0_output(request_id),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[21, 22, 23],
        )
        events = await h.settle()

        assert not [event for event in events if event.type == "error"], types(events)
        assert len(h.port.submissions) == 2, "the stop token did not schedule the next unit"
        silence = h.port.submissions[1]
        assert silence.already_submitted is True
        assert h.runner.model_state.continuation_units == 1

        # The audio-side trigger for the same unit must not schedule a second
        # silence unit when the audio finally arrives.
        await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True, finished=True)
        )
        assert len(h.port.submissions) == 2, "the audio-emit trigger double-scheduled a unit"
        assert h.runner.model_state.continuation_units == 1
        assert h.session.active_response_id is not None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_audio_mode_parity_keeps_the_audio_side_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default mode: the Stage-0 segment finish alone schedules nothing.

    Bit-parity with the historical behaviour: only the TTS segment end (audio
    emit) plans the next unit.
    """
    monkeypatch.delenv(CONTINUE_ON_ENV, raising=False)
    h = await _active_response_harness()
    try:
        request_id = h.stage0_request_id()
        _make_continuation_due_now(h)

        h.deliver(
            spoken_stage0_output(request_id),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[21, 22, 23],
        )
        await h.settle()
        assert len(h.port.submissions) == 1, "stop token scheduled a unit in audio mode"

        await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True, finished=True)
        )
        assert len(h.port.submissions) == 2, "the audio-emit trigger did not schedule the unit"
        assert h.runner.model_state.continuation_units == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stop_token_mode_preserves_client_audio_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit k+1's append may be submitted before unit k's audio; order holds.

    The correctness red line of the decoupling: submission order and emission
    order are different queues. The test drives two spoken units whose appends
    are submitted at their stop tokens, with each unit's audio delivered only
    afterwards, and asserts the client still hears unit k's full audio before
    unit k+1's first delta.
    """
    monkeypatch.setenv(CONTINUE_ON_ENV, "stop_token")
    h = await _active_response_harness()
    try:
        request_id = h.stage0_request_id()

        # Unit 2's stop token: its continuation (unit 3's append) is submitted
        # before unit 2's audio exists.
        _make_continuation_due_now(h)
        h.deliver(
            spoken_stage0_output(request_id),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[21, 22, 23],
        )
        await h.settle()
        assert len(h.port.submissions) == 2

        # Unit 2's audio arrives late; the client sees its delta now. (The
        # data plane slices cumulative audio, so each unit's snapshot must
        # grow past the previous unit's offset.)
        events_unit2 = await h.deliver_and_settle(tts_output(request_id, samples=48000, text="second"))
        delta_unit2 = [event for event in events_unit2 if event.type == "response.output_audio.delta"]
        assert len(delta_unit2) == 1

        # Unit 3's stop token schedules unit 4's append while unit 3's audio
        # is also still pending.
        _make_continuation_due_now(h)
        h.deliver(
            spoken_stage0_output(request_id),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[31, 32, 33],
        )
        await h.settle()
        assert len(h.port.submissions) == 3

        events_unit3 = await h.deliver_and_settle(tts_output(request_id, samples=72000, text="third"))
        delta_unit3 = [event for event in events_unit3 if event.type == "response.output_audio.delta"]
        assert len(delta_unit3) == 1

        # Emission order follows delivery order, which follows per-session
        # stage processing order -- never the append submission order.
        assert h.events.index(delta_unit2[0]) < h.events.index(delta_unit3[0])
        assert not [event for event in events_unit3 if event.type == "error"], types(events_unit3)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stop_token_mode_listen_segment_keeps_listen_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listen decision segment is not a stop-token trigger.

    The listen path already continued on Stage 0 output before this switch
    existed; the hook must stay out of its way (no second unit for one
    segment).
    """
    from tests.engine.duplex.test_session_runner import listen_output

    monkeypatch.setenv(CONTINUE_ON_ENV, "stop_token")
    h = await _active_response_harness()
    try:
        request_id = h.stage0_request_id()
        submissions_before = len(h.port.submissions)

        listen = listen_output(request_id)
        listen.finished = False
        await h.deliver_and_settle(
            listen,
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
            segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
        )
        # Exactly the listen path's one continuation, not two.
        assert len(h.port.submissions) == submissions_before + 1
        assert h.runner.model_state.continuation_units == 1
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Unit depth                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_depth_two_plans_second_continuation_while_first_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parked first continuation append must not block planning at depth 2."""
    monkeypatch.setenv(DEPTH_ENV, "2")
    h = await _active_response_harness()
    try:
        _make_continuation_due_now(h)
        gate = asyncio.Event()
        h.port.submit_gate = gate
        h.port.submit_started.clear()
        kwargs = _continuation_kwargs(h)
        payload = h.runner.model.silence_unit_payload()

        first = asyncio.ensure_future(h.runner._schedule_silence_continuation(payload, **kwargs))
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
        assert len(h.runner._silence_in_flight()) == 1

        second = asyncio.ensure_future(h.runner._schedule_silence_continuation(payload, **kwargs))
        scheduled = await asyncio.wait_for(asyncio.shield(second), timeout=1.0)
        assert scheduled is True, "depth 2 did not plan the second unit while the first submits"
        assert len(h.runner._silence_in_flight()) == 2

        gate.set()
        assert await asyncio.wait_for(first, timeout=2.0) is True
        assert await asyncio.wait_for(second, timeout=2.0) is True
        await h.settle()
        assert len(h.port.submissions) == 3  # user append + two silence units
        # The scheduler entry was called directly (not through the model
        # channel), so the continuation-unit counter stays untouched; the
        # in-flight window is the observable and it drains to empty.
        assert h.runner._silence_in_flight() == []
    finally:
        h.port.submit_gate = None
        await close_harness(h)


@pytest.mark.asyncio
async def test_depth_one_still_waits_for_the_parked_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default depth keeps the historical single-flight wait."""
    monkeypatch.delenv(DEPTH_ENV, raising=False)
    h = await _active_response_harness()
    try:
        _make_continuation_due_now(h)
        gate = asyncio.Event()
        h.port.submit_gate = gate
        h.port.submit_started.clear()
        kwargs = _continuation_kwargs(h)
        payload = h.runner.model.silence_unit_payload()

        first = asyncio.ensure_future(h.runner._schedule_silence_continuation(payload, **kwargs))
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)

        second = asyncio.ensure_future(h.runner._schedule_silence_continuation(payload, **kwargs))
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await asyncio.wait_for(asyncio.shield(second), timeout=0.3)
        assert len(h.runner._silence_in_flight()) == 1

        gate.set()
        assert await asyncio.wait_for(first, timeout=2.0) is True
        assert await asyncio.wait_for(second, timeout=2.0) is True
        await h.settle()
        assert len(h.port.submissions) == 3
    finally:
        h.port.submit_gate = None
        await close_harness(h)


@pytest.mark.asyncio
async def test_real_append_invalidates_a_planned_continuation_at_any_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real (user) append accepted while a continuation sleeps supersedes it.

    The generation check replaced the raw anchor-timestamp equality when the
    depth window was introduced; this pins the invariant it protected: the
    stale unit is called off before submission, at every depth.
    """
    monkeypatch.setenv(DEPTH_ENV, "2")
    h = await _active_response_harness()
    try:
        model_state = h.runner.model_state
        # A continuation planned 0.4 s into the current anchor's period.
        model_state.last_native_submit_monotonic = time.monotonic() - 0.4
        model_state.silence_deadline_monotonic = None
        kwargs = _continuation_kwargs(h)

        planned = asyncio.ensure_future(
            h.runner._schedule_silence_continuation(h.runner.model.silence_unit_payload(), **kwargs)
        )
        await asyncio.sleep(0)

        # A real append is accepted while the continuation still sleeps: it
        # re-anchors the chain and bumps the generation.
        model_state.last_native_submit_monotonic = time.monotonic()
        model_state.silence_deadline_monotonic = None
        model_state.native_input_generation += 1

        await asyncio.wait_for(planned, timeout=3.0)
        await h.settle()
        # Called off before submission: no silence unit reached the stage.
        assert len(h.port.submissions) == 1
        assert h.runner._silence_in_flight() == []
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Flag parsing                                                                #
# --------------------------------------------------------------------------- #


def test_continue_on_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.engine.duplex.config import duplex_continue_on_stop_token

    monkeypatch.delenv(CONTINUE_ON_ENV, raising=False)
    assert duplex_continue_on_stop_token() is False

    monkeypatch.setenv(CONTINUE_ON_ENV, "audio")
    assert duplex_continue_on_stop_token() is False

    monkeypatch.setenv(CONTINUE_ON_ENV, "stop_token")
    assert duplex_continue_on_stop_token() is True

    monkeypatch.setenv(CONTINUE_ON_ENV, " Stop-Token ")
    assert duplex_continue_on_stop_token() is True

    monkeypatch.setenv(CONTINUE_ON_ENV, "bogus")
    assert duplex_continue_on_stop_token() is False


def test_unit_depth_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.engine.duplex.config import duplex_unit_depth

    monkeypatch.delenv(DEPTH_ENV, raising=False)
    assert duplex_unit_depth() == 1

    monkeypatch.setenv(DEPTH_ENV, "2")
    assert duplex_unit_depth() == 2

    monkeypatch.setenv(DEPTH_ENV, "0")
    assert duplex_unit_depth() == 1

    monkeypatch.setenv(DEPTH_ENV, "-3")
    assert duplex_unit_depth() == 1

    monkeypatch.setenv(DEPTH_ENV, "64")
    assert duplex_unit_depth() == 8

    monkeypatch.setenv(DEPTH_ENV, "not-a-number")
    assert duplex_unit_depth() == 1

    monkeypatch.setenv(DEPTH_ENV, "")
    assert duplex_unit_depth() == 1
