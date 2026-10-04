# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tracking and acknowledgements of input-clocked sessions, with fake stage progress."""

from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexOutputContext, DuplexRequestIdentity
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    DuplexUnitDecision,
    DuplexUnitOutputs,
)
from vllm_omni.engine.duplex.session.input_clock import (
    DEFAULT_UNIT_MAX_AGE_S,
    DEFAULT_UNIT_TIMEOUT_S,
    InputClock,
    InputClockUnit,
    StageProgress,
    check_input_clock_supported,
    check_input_clock_unchanged,
    input_clock_lease_idle_s,
    input_clocked,
    unit_max_age_s,
    unit_timeout_s,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.protocol.duplex.errors import REALTIME_ERROR_TYPES_BY_CODE
from vllm_omni.protocol.duplex.events import InputProcessed

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

APPEND = "input_audio_buffer.append"
COMMIT = "input_audio_buffer.commit"
SESSION = "sess-clock"


class _DefaultHooksPlugin(MiniCPMO45DuplexPlugin):
    """A real plugin with the contract's default input-clock hooks (whatever the model overrides)."""

    unit_decision = DuplexModelPlugin.unit_decision
    unit_output_complete = DuplexModelPlugin.unit_output_complete


@dataclass(frozen=True)
class _FramesOutput:
    """A final-stage output reporting how many audio frames it carries."""

    frames: int


class _FrameCoveragePlugin(_DefaultHooksPlugin):
    """A final stage that reports audio frames and never marks unit ends, like a frame model.

    Every Stage-0 submission yields one frame, so the epoch's frame count
    covers the units up to that ordinal (``state`` starts over in each epoch).
    """

    def unit_output_complete(
        self,
        *,
        unit: DuplexUnitOutputs | None,
        final_stage_id: int,
        new_output: object | None,
        new_context: DuplexOutputContext | None,
        state: dict[str, object],
        runtime_config: object,
    ) -> bool:
        del final_stage_id, new_context, runtime_config
        previous = state.get("frames", 0)
        frames = previous if isinstance(previous, int) else 0
        if isinstance(new_output, _FramesOutput):
            frames += new_output.frames
            state["frames"] = frames
        return unit is not None and frames >= unit.ordinal


def _plugin(cls: type[MiniCPMO45DuplexPlugin] = _DefaultHooksPlugin) -> MiniCPMO45DuplexPlugin:
    return cls(lambda *args: None)


def _clock(
    plugin: DuplexModelPlugin | None = None, *, now: list[float] | None = None
) -> tuple[InputClock, list[InputProcessed]]:
    sent: list[InputProcessed] = []
    timer = now if now is not None else [0.0]
    clock = InputClock(
        emit=lambda events: sent.extend(e for e in events if isinstance(e, InputProcessed)),
        plugin=plugin if plugin is not None else _plugin(),
        runtime_config=dict,
        clock=lambda: timer[0],
    )
    return clock, sent


def _context(*, final_stage_id: int, segment_finished: bool, epoch: int) -> DuplexOutputContext:
    return DuplexOutputContext(
        identity=DuplexRequestIdentity(session_id=SESSION, fence=DuplexFence(SESSION, epoch=epoch)),
        final_stage_id=final_stage_id,
        segment_finished=segment_finished,
    )


def _progress(
    clock: InputClock,
    *,
    stage_id: int,
    final_stage_id: int = 1,
    segment_finished: bool = True,
    decision: str | None = None,
    ends_unit: bool = True,
    output: object = None,
    epoch: int = 0,
) -> None:
    final = stage_id >= final_stage_id
    clock.on_stage_progress(
        StageProgress(
            stage_id=stage_id,
            final_stage_id=final_stage_id,
            epoch=epoch,
            segment_finished=segment_finished,
            decision=DuplexUnitDecision(label=decision, ends_unit=ends_unit) if decision is not None else None,
            output=output if final else None,
            context=_context(final_stage_id=final_stage_id, segment_finished=segment_finished, epoch=epoch)
            if final
            else None,
        )
    )


def _append(
    clock: InputClock, *, ms: int = 1000, units: int = 1, epoch: int = 0, accepted: bool = True
) -> list[InputClockUnit]:
    """One append of ``ms`` of audio that submits ``units`` units (accepted by Stage 0 unless told otherwise)."""
    ack = clock.begin_input(APPEND)
    clock.note_input_audio(ack, samples=16 * ms, sample_rate_hz=16000)
    created = [clock.new_unit(input_us=ms * 1000 // max(units, 1), epoch=epoch) for _ in range(units)]
    if accepted:
        for unit in created:
            clock.unit_submitted(unit)
    clock.end_input(ack)
    return created


def _decisions(sent: list[InputProcessed]) -> list[list[tuple[object, object]]]:
    return [[(unit["decision"], unit.get("reason")) for unit in event.units] for event in sent]


# --------------------------------------------------------------------------- #
# Configuration and contract                                                  #
# --------------------------------------------------------------------------- #


def test_opt_in_and_timeout_parsing() -> None:
    # The parse helpers fall back to the defaults; a session with an invalid value is refused
    # (``check_input_clock_supported``, see the next test).
    assert input_clocked({"clock": "input"})
    assert not input_clocked({"clock": "wall"}) and not input_clocked(None)
    assert unit_timeout_s({"input_clock_unit_timeout_s": 3}) == 3.0
    assert unit_timeout_s({"input_clock_unit_timeout_s": True}) == DEFAULT_UNIT_TIMEOUT_S
    assert unit_max_age_s({}) == DEFAULT_UNIT_MAX_AGE_S == 60.0
    assert unit_max_age_s({"input_clock_unit_max_s": 120}) == 120.0
    assert unit_max_age_s({"input_clock_unit_max_s": -1}) == DEFAULT_UNIT_MAX_AGE_S


@pytest.mark.parametrize("value", [0, -1, "30", True, float("nan"), float("inf")])
@pytest.mark.parametrize("key", ["input_clock_unit_timeout_s", "input_clock_unit_max_s"])
def test_an_invalid_timeout_is_refused_instead_of_replaced_by_the_default(key: str, value: object) -> None:
    plugin = _plugin()
    plugin.supports_input_clock = True
    with pytest.raises(DuplexRuntimeConfigError) as excinfo:
        check_input_clock_supported(plugin, {"clock": "input", key: value})
    assert excinfo.value.code == "invalid_duplex_runtime_config"
    check_input_clock_supported(plugin, {"clock": "input", key: 0.5})
    check_input_clock_supported(plugin, {"clock": "input", key: None})  # null: the default
    check_input_clock_supported(plugin, {key: value})  # inert without the clock


def test_a_plugin_that_has_not_opted_in_refuses_the_input_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    assert REALTIME_ERROR_TYPES_BY_CODE["input_clock_unsupported"] == "invalid_request_error"
    assert DuplexModelPlugin.supports_input_clock is False
    plugin = _plugin()
    monkeypatch.setattr(plugin, "supports_input_clock", False)
    with pytest.raises(DuplexRuntimeConfigError) as excinfo:
        check_input_clock_supported(plugin, {"clock": "input"})
    assert excinfo.value.code == "input_clock_unsupported"
    check_input_clock_supported(plugin, {"clock": None})
    monkeypatch.setattr(plugin, "supports_input_clock", True)
    check_input_clock_supported(plugin, {"clock": "input"})


def test_the_clock_and_its_timeouts_are_fixed_at_creation() -> None:
    assert REALTIME_ERROR_TYPES_BY_CODE["input_clock_update_unsupported"] == "invalid_request_error"
    clocked = {"clock": "input"}
    check_input_clock_unchanged(clocked, {"clock": "input", "input_clock_unit_timeout_s": DEFAULT_UNIT_TIMEOUT_S})
    check_input_clock_unchanged({}, {"clock": None, "overlap_policy": "barge_in"})
    # Without the clock, the timeout keys are inert and may change.
    check_input_clock_unchanged({}, {"input_clock_unit_timeout_s": 3, "input_clock_unit_max_s": 120})
    for current, candidate in [
        ({}, clocked),
        (clocked, {"clock": None}),
        (clocked, {**clocked, "input_clock_unit_timeout_s": 3}),
        (clocked, {**clocked, "input_clock_unit_max_s": 120}),
    ]:
        with pytest.raises(DuplexRuntimeConfigError) as excinfo:
            check_input_clock_unchanged(current, candidate)
        assert excinfo.value.code == "input_clock_update_unsupported"


def test_the_engine_lease_idle_window_is_capped() -> None:
    assert input_clock_lease_idle_s(600.0, 300.0) == 600.0
    assert input_clock_lease_idle_s(30.0, 300.0) == 30.0
    assert input_clock_lease_idle_s(86_400.0, 300.0) == 600.0, "a client cannot hold its slot for a day"
    assert input_clock_lease_idle_s(86_400.0, 3600.0) == 3600.0, "a deploy may allow longer"
    assert input_clock_lease_idle_s(600.0, None) is None, "a deploy without idle expiry expires nothing"


def test_the_acknowledgement_has_only_its_own_wire_fields() -> None:
    clock, sent = _clock()
    _append(clock, ms=200, units=0)

    (ack,) = sent
    assert ack.type == "input_audio_buffer.processed"
    assert (ack.audio_end_ms, ack.unit_end_ms, ack.units) == (200, 0, ())
    assert (ack.input_index, ack.first_input_index, ack.trigger) == (1, None, APPEND)
    own = {f.name for f in fields(InputProcessed)} - {"session_id", "epoch", "event_id"}
    assert own == {"audio_end_ms", "unit_end_ms", "input_index", "first_input_index", "trigger", "units"}
    assert InputProcessed.optional_wire_fields == frozenset({"first_input_index"})


# --------------------------------------------------------------------------- #
# Acknowledgement order                                                       #
# --------------------------------------------------------------------------- #


def test_an_input_that_only_buffers_audio_is_acknowledged_once_prior_acknowledgements_are_out() -> None:
    clock, sent = _clock()
    _append(clock)  # unit 0 in flight
    _append(clock, ms=200, units=0)  # buffers only: its watermark is already reached ...
    assert sent == [], "... but it must not overtake the acknowledgement before it"

    _progress(clock, stage_id=0, decision="listen")

    assert [e.input_index for e in sent] == [1, 2]
    assert sent[1].units == () and sent[1].audio_end_ms == 1200


def test_one_append_that_submits_two_units_gets_one_acknowledgement_after_both() -> None:
    clock, sent = _clock()
    _append(clock, ms=2000, units=2)
    _progress(clock, stage_id=0, decision="listen")
    assert sent == [], "the second unit is still in flight"

    _progress(clock, stage_id=0, decision="listen")

    (ack,) = sent
    assert ack.units == ({"end_ms": 1000, "decision": "listen"}, {"end_ms": 2000, "decision": "listen"})
    assert (ack.unit_end_ms, ack.audio_end_ms) == (2000, 2000)


def test_a_listen_unit_is_acknowledged_after_its_stage0_decision() -> None:
    clock, sent = _clock()
    _append(clock)
    assert sent == []

    _progress(clock, stage_id=0, decision="listen")

    assert [e.units for e in sent] == [({"end_ms": 1000, "decision": "listen"},)]
    assert sent[0].unit_end_ms == 1000


def test_a_speaking_unit_waits_for_the_final_stage_segment_end_by_default() -> None:
    clock, sent = _clock()
    _append(clock)
    _progress(clock, stage_id=0)  # forwarded: speaking
    _progress(clock, stage_id=1, segment_finished=False)
    assert sent == []

    _progress(clock, stage_id=1, segment_finished=True)

    assert sent[0].units == ({"end_ms": 1000, "decision": "speak"},)


def test_an_intermediate_stage_finishing_does_not_complete_a_unit_that_still_owes_audio() -> None:
    clock, sent = _clock()
    _append(clock)
    _progress(clock, stage_id=0, final_stage_id=2)
    _progress(clock, stage_id=1, final_stage_id=2, segment_finished=True)  # text stage done
    assert sent == []

    _progress(clock, stage_id=2, final_stage_id=2, segment_finished=True)

    assert sent[0].units == ({"end_ms": 1000, "decision": "speak"},)


def test_upstream_segment_ends_are_credited_to_the_speaking_units_in_order() -> None:
    """The stage before the final one ends unit 2's segment while unit 1 still waits for its audio."""

    class _NeedsUpstream(_DefaultHooksPlugin):
        def unit_output_complete(self, *, unit, final_stage_id, new_output, new_context, state, runtime_config):
            del final_stage_id, new_output, new_context, state, runtime_config
            return unit is not None and unit.upstream_segment_finished and unit.final_segment_finished

    clock, sent = _clock(_plugin(_NeedsUpstream))
    _append(clock)
    _append(clock)
    _progress(clock, stage_id=0, final_stage_id=2)
    _progress(clock, stage_id=0, final_stage_id=2)
    _progress(clock, stage_id=1, final_stage_id=2)  # unit 1's text segment
    _progress(clock, stage_id=1, final_stage_id=2)  # unit 2's text segment

    _progress(clock, stage_id=2, final_stage_id=2)
    _progress(clock, stage_id=2, final_stage_id=2)

    assert [e.input_index for e in sent] == [1, 2], "unit 2 must not wait for a third upstream segment end"


def test_acknowledgements_keep_input_order_and_units_complete_as_a_prefix() -> None:
    clock, sent = _clock()
    _append(clock)  # unit 0: will speak
    _progress(clock, stage_id=0)
    _append(clock)  # unit 1: listens before unit 0 finished speaking
    _progress(clock, stage_id=0, decision="listen")
    assert sent == [], "unit 1 is complete but unit 0 is not"

    _progress(clock, stage_id=1, segment_finished=True)

    assert [e.input_index for e in sent] == [1, 2]
    # Each acknowledgement reports only the units created up to its own input.
    assert (sent[0].units, sent[0].unit_end_ms) == (({"end_ms": 1000, "decision": "speak"},), 1000)
    assert (sent[1].units, sent[1].unit_end_ms) == (({"end_ms": 2000, "decision": "listen"},), 2000)


def test_a_later_stage_decision_that_ends_the_pipeline_completes_the_unit() -> None:
    """A silent second stage (no TTS) ends a committed turn mid-pipeline."""
    clock, sent = _clock()
    _append(clock)
    _progress(clock, stage_id=0, final_stage_id=3)
    _progress(clock, stage_id=1, final_stage_id=3, decision="listen")

    assert sent[0].units == ({"end_ms": 1000, "decision": "listen"},)


def test_a_side_channel_decision_keeps_the_unit_open_and_one_output_can_cover_several_units() -> None:
    clock, sent = _clock(_plugin(_FrameCoveragePlugin))
    for _ in range(3):
        _append(clock, ms=80)
        _progress(clock, stage_id=0, final_stage_id=2, decision="listen", ends_unit=False)
    assert sent == []

    _progress(clock, stage_id=2, final_stage_id=2, segment_finished=False, output=_FramesOutput(frames=1))
    assert [e.input_index for e in sent] == [1]
    _progress(clock, stage_id=2, final_stage_id=2, segment_finished=False, output=_FramesOutput(frames=2))

    assert [e.input_index for e in sent] == [1, 2, 3]
    assert [u["decision"] for e in sent for u in e.units] == ["listen"] * 3


def test_the_plugin_sees_each_units_position_in_its_epoch_and_a_fresh_state_after_a_cancel() -> None:
    """A frame count kept in ``state`` stays aligned after a cancel and after a unit that never reached Stage 0."""
    seen: list[tuple[int, int, int]] = []

    class _Recording(_FrameCoveragePlugin):
        def unit_output_complete(self, *, unit, **kwargs):
            if unit is not None:
                seen.append((unit.index, unit.epoch, unit.ordinal))
            return super().unit_output_complete(unit=unit, **kwargs)

    clock, sent = _clock(_plugin(_Recording))
    _append(clock, epoch=0)  # unit 0 speaks, then a barge-in before any of its frames
    _progress(clock, stage_id=0, decision="listen", ends_unit=False, epoch=0)
    clock.cancel_open("barge_in", before_epoch=1)
    (dropped,) = _append(clock, epoch=1, accepted=False)
    clock.append_finished(dropped)  # never reached Stage 0: no ordinal
    for _ in range(2):
        _append(clock, epoch=1)
        _progress(clock, stage_id=0, decision="listen", ends_unit=False, epoch=1)

    _progress(clock, stage_id=1, segment_finished=False, output=_FramesOutput(frames=1), epoch=1)
    assert [e.input_index for e in sent] == [1, 2, 3], "the new epoch's first frame covers its first unit"
    _progress(clock, stage_id=1, segment_finished=False, output=_FramesOutput(frames=1), epoch=1)

    assert [e.input_index for e in sent] == [1, 2, 3, 4]
    assert {entry for entry in seen if entry[1] == 1} == {(2, 1, 1), (3, 1, 2)}, "(index, epoch, ordinal)"


def test_a_final_output_with_no_speaking_unit_is_ignored() -> None:
    clock, sent = _clock()
    _progress(clock, stage_id=1, segment_finished=True)
    _progress(clock, stage_id=0)  # a segment end with nothing in flight
    assert sent == []


def test_a_segment_end_that_overtakes_the_acceptance_callback_decides_the_unit_in_submission() -> None:
    clock, sent = _clock()
    (unit,) = _append(clock, accepted=False)  # Stage 0 has it; the callback has not run yet

    _progress(clock, stage_id=0, decision="listen")
    clock.unit_submitted(unit)  # the callback, late

    assert _decisions(sent) == [[("listen", None)]]
    (second,) = _append(clock)
    assert second.ordinal == 2, "the late callback does not take a second ordinal"


# --------------------------------------------------------------------------- #
# Deferred committed turns                                                    #
# --------------------------------------------------------------------------- #


def test_a_deferred_committed_turn_is_counted_before_it_is_submitted() -> None:
    clock, sent = _clock()
    ack = clock.begin_input(COMMIT)
    clock.reserve_unit("turn-a")
    clock.end_input(ack)
    _append(clock, ms=200, units=0)
    assert sent == []

    clock.unit_submitted(clock.new_unit(input_us=500_000, epoch=0, turn="turn-a"))
    _progress(clock, stage_id=0, decision="listen")

    assert [e.trigger for e in sent] == [COMMIT, APPEND]
    assert sent[0].units == ({"end_ms": 500, "decision": "listen"},)


def test_a_reserved_turn_is_claimed_only_by_its_own_submission() -> None:
    clock, _ = _clock()
    ack = clock.begin_input(COMMIT)
    clock.reserve_unit("turn-a")
    clock.end_input(ack)

    other = clock.new_unit(input_us=100_000, epoch=0, turn="turn-b")  # another committed turn
    assert other.seq == 2, "a different turn must not take the reserved slot"
    clock.rekey_placeholder("turn-a", "turn-c")  # merged into a later commit
    assert clock.placeholder_turns() == ["turn-c"]
    claimed = clock.new_unit(input_us=500_000, epoch=0, turn="turn-c")
    assert claimed.seq == 1
    assert clock.placeholder_turns() == []


def test_a_second_deferred_commit_merges_into_the_reserved_turn() -> None:
    clock, sent = _clock()
    for turn in ("turn-a", "turn-b"):
        ack = clock.begin_input(COMMIT)
        clock.reserve_unit(turn)
        clock.end_input(ack)
    assert clock.placeholder_turns() == ["turn-b"], "one submission, one slot"

    clock.unit_submitted(clock.new_unit(input_us=500_000, epoch=0, turn="turn-b"))
    _progress(clock, stage_id=0, decision="listen")

    assert [e.input_index for e in sent] == [1, 2]


def test_a_dropped_deferred_turn_releases_its_slot_as_cancelled() -> None:
    clock, sent = _clock()
    ack = clock.begin_input(COMMIT)
    clock.reserve_unit("turn-a")
    clock.end_input(ack)
    clock.release_placeholder("turn-other", "input_cleared")
    assert sent == []

    clock.release_placeholder("turn-a", "input_cleared")

    assert _decisions(sent) == [[("cancelled", "input_cleared")]]
    assert not clock.has_open_units()


def test_a_reserved_turn_submitted_after_its_slot_timed_out_takes_the_settled_slot() -> None:
    """Submitted by the end of the response: its output stays its own, no later input waits for it."""
    now = [0.0]
    clock, sent = _clock(now=now)
    ack = clock.begin_input(COMMIT)
    clock.reserve_unit("turn-a")
    clock.end_input(ack)
    now[0] = DEFAULT_UNIT_TIMEOUT_S
    assert clock.expire() is True
    assert _decisions(sent) == [[("timed_out", "no_progress")]]

    late = clock.new_unit(input_us=500_000, epoch=0, turn="turn-a", from_input=False)
    clock.unit_submitted(late)
    assert (late.seq, late.ordinal) == (1, 1) and not clock.has_open_units()
    _append(clock, ms=200, units=0)  # buffers only: nothing to wait for
    assert [e.input_index for e in sent] == [1, 2]

    (second,) = _append(clock)
    _progress(clock, stage_id=0, decision="listen")  # the late turn's segment end: its own
    assert len(sent) == 2
    _progress(clock, stage_id=0, decision="listen")
    assert second.ordinal == 2 and [e.input_index for e in sent] == [1, 2, 3]
    assert _decisions(sent)[2] == [("listen", None)]


def test_a_released_turn_that_a_client_input_submits_is_that_inputs_unit() -> None:
    clock, sent = _clock()
    ack = clock.begin_input(COMMIT)
    clock.reserve_unit("turn-a")
    clock.end_input(ack)
    clock.release_placeholder("turn-a", "output_audio_buffer_clear")  # its response was cancelled
    assert _decisions(sent) == [[("cancelled", "output_audio_buffer_clear")]]

    ack = clock.begin_input("response.create")
    unit = clock.new_unit(input_us=500_000, epoch=1, turn="turn-a", from_input=True)
    clock.unit_submitted(unit)
    clock.end_input(ack)
    assert unit.seq == 2 and sent[-1].input_index == 1, "response.create waits for the turn it submitted"

    _progress(clock, stage_id=0, decision="listen", epoch=1)
    assert [e.input_index for e in sent] == [1, 2] and _decisions(sent)[1] == [("listen", None)]


# --------------------------------------------------------------------------- #
# Settling                                                                    #
# --------------------------------------------------------------------------- #


def test_a_unit_that_never_reached_stage0_is_reported_dropped() -> None:
    clock, sent = _clock()
    (unit,) = _append(clock, accepted=False)
    clock.append_finished(unit)  # called off before submission

    assert sent[0].units == ({"end_ms": 1000, "decision": "dropped"},)


def test_a_submitted_unit_is_not_settled_by_its_successful_append() -> None:
    clock, sent = _clock()
    (unit,) = _append(clock)
    clock.append_finished(unit)
    assert sent == []


def test_a_unit_accepted_after_its_epoch_was_cancelled_takes_no_ordinal() -> None:
    clock, sent = _clock()
    (unit,) = _append(clock, accepted=False)
    clock.cancel_open("barge_in", before_epoch=1)
    clock.unit_submitted(unit)  # the acceptance callback of the cancelled epoch, late

    assert unit.ordinal is None
    (second,) = _append(clock, epoch=1)
    assert second.ordinal == 1
    _progress(clock, stage_id=0, decision="listen", epoch=1)
    assert _decisions(sent) == [[("cancelled", "barge_in")], [("listen", None)]]


def test_cancel_settles_open_units_of_the_cancelled_epoch_and_ignores_their_late_output() -> None:
    clock, sent = _clock()
    _append(clock, epoch=0)
    _progress(clock, stage_id=0, epoch=0)  # speaking
    _append(clock, epoch=0)  # still undecided
    ack = clock.begin_input(COMMIT)
    clock.reserve_unit("turn-a")  # a deferred turn survives the cancel unless it was dropped
    clock.end_input(ack)

    clock.cancel_open("client_cancelled", before_epoch=1)

    assert [e.input_index for e in sent] == [1, 2]
    assert _decisions(sent) == [[("cancelled", "client_cancelled")], [("cancelled", "client_cancelled")]]
    assert clock.placeholder_turns() == ["turn-a"]
    _progress(clock, stage_id=0, decision="listen", epoch=0)  # late output of the cancelled epoch
    _progress(clock, stage_id=1, segment_finished=True, epoch=0)
    assert len(sent) == 2

    clock.unit_submitted(clock.new_unit(input_us=1_000_000, epoch=1, turn="turn-a"))
    _progress(clock, stage_id=0, decision="listen", epoch=1)
    assert [e.input_index for e in sent] == [1, 2, 3]
    assert sent[2].units == ({"end_ms": 3000, "decision": "listen"},)


def test_close_acknowledges_every_owed_input_with_one_event_and_then_stays_silent() -> None:
    clock, sent = _clock()
    _append(clock)
    _progress(clock, stage_id=0)  # speaking
    in_progress = clock.begin_input(APPEND)  # an input still being handled
    clock.new_unit(input_us=1_000_000, epoch=0)

    clock.close("client_close")

    (ack,) = sent
    assert (ack.first_input_index, ack.input_index, ack.trigger) == (1, 2, APPEND)
    assert _decisions(sent) == [[("aborted", "client_close")] * 2]
    clock.end_input(in_progress)
    _append(clock)
    _progress(clock, stage_id=1, segment_finished=True)
    assert len(sent) == 1, "nothing follows the teardown"


def test_close_with_one_owed_input_sends_a_plain_acknowledgement() -> None:
    clock, sent = _clock()
    _append(clock)

    clock.close("idle_ttl_expired")

    (ack,) = sent
    assert (ack.first_input_index, ack.input_index) == (None, 1)
    assert _decisions(sent) == [[("aborted", "idle_ttl_expired")]]


# --------------------------------------------------------------------------- #
# Timeouts                                                                    #
# --------------------------------------------------------------------------- #


def test_the_liveness_valve_settles_a_unit_after_a_silent_pipeline() -> None:
    now = [0.0]
    clock, sent = _clock(now=now)
    _append(clock)
    _progress(clock, stage_id=0)
    now[0] = DEFAULT_UNIT_TIMEOUT_S - 1
    assert clock.expire() is False

    now[0] = DEFAULT_UNIT_TIMEOUT_S + 1
    assert clock.expire() is True
    assert sent[0].units == ({"end_ms": 1000, "decision": "timed_out", "reason": "no_progress"},)
    assert not clock.has_open_units()


def test_client_input_does_not_hold_off_the_liveness_valve_but_an_idle_pipeline_starts_afresh() -> None:
    now = [0.0]
    clock, sent = _clock(now=now)
    _append(clock)
    now[0] = DEFAULT_UNIT_TIMEOUT_S - 1
    _append(clock)  # more input while the pipeline is silent

    now[0] = DEFAULT_UNIT_TIMEOUT_S
    assert clock.expire() is True
    assert _decisions(sent) == [[("timed_out", "no_progress")]]

    # Later, with nothing in flight for a long time, a new unit gets a whole window.
    now[0] = 1000.0
    _progress(clock, stage_id=0, decision="listen")  # unit 1's segment end, late
    _progress(clock, stage_id=0, decision="listen")  # unit 2's
    assert [e.input_index for e in sent] == [1, 2]
    now[0] = 5000.0
    _append(clock)
    now[0] = 5000.0 + DEFAULT_UNIT_TIMEOUT_S - 1
    assert clock.expire() is False


def test_the_maximum_unit_age_settles_a_unit_that_keeps_producing_but_never_completes() -> None:
    now = [0.0]
    clock, sent = _clock(now=now)
    _append(clock)
    _progress(clock, stage_id=0)  # speaking
    for second in range(1, int(DEFAULT_UNIT_MAX_AGE_S)):
        now[0] = float(second)
        clock.note_progress()
        _progress(clock, stage_id=1, segment_finished=False)  # streaming, never finished
        assert clock.expire() is False

    now[0] = DEFAULT_UNIT_MAX_AGE_S
    assert clock.expire() is True

    assert sent[0].units == ({"end_ms": 1000, "decision": "timed_out", "reason": "max_age"},)


def test_a_late_segment_end_of_a_timed_out_unit_is_not_credited_to_the_next_unit() -> None:
    now = [0.0]
    clock, sent = _clock(now=now)
    _append(clock)
    now[0] = DEFAULT_UNIT_TIMEOUT_S
    assert clock.expire() is True
    _append(clock)

    _progress(clock, stage_id=0)  # unit 1's segment end, late: it would have spoken
    assert len(sent) == 1, "unit 2 is still undecided"
    _progress(clock, stage_id=0)  # unit 2 speaks too
    _progress(clock, stage_id=1, segment_finished=True)  # unit 1's audio, late: still its own
    assert len(sent) == 1
    _progress(clock, stage_id=1, segment_finished=True)

    assert [e.input_index for e in sent] == [1, 2]
    assert _decisions(sent) == [[("timed_out", "no_progress")], [("speak", None)]]


def test_late_final_stage_output_of_a_timed_out_speaking_unit_stays_its_own() -> None:
    now = [0.0]
    clock, sent = _clock(now=now)
    _append(clock)
    _progress(clock, stage_id=0)  # unit 1 speaks
    now[0] = DEFAULT_UNIT_MAX_AGE_S
    clock.note_progress()
    assert clock.expire() is True
    assert _decisions(sent) == [[("timed_out", "max_age")]]
    _append(clock)
    _progress(clock, stage_id=0)  # unit 2 speaks

    _progress(clock, stage_id=1, segment_finished=True)  # the end of unit 1's audio
    assert len(sent) == 1, "unit 2's audio has not ended"
    _progress(clock, stage_id=1, segment_finished=True)

    assert [e.input_index for e in sent] == [1, 2] and _decisions(sent)[1] == [("speak", None)]


# --------------------------------------------------------------------------- #
# Plugin hook failures                                                        #
# --------------------------------------------------------------------------- #


def test_a_raising_completion_hook_leaves_the_unit_to_the_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.engine.duplex.session import input_clock as input_clock_module

    logged: list[str] = []
    monkeypatch.setattr(input_clock_module.logger, "exception", lambda msg, *a, **k: logged.append(msg % a))

    class _Raising(_DefaultHooksPlugin):
        def unit_output_complete(self, **kwargs):
            raise RuntimeError("plugin bug")

    now = [0.0]
    clock, sent = _clock(_plugin(_Raising), now=now)
    _append(clock)
    _progress(clock, stage_id=0)  # speaking: the hook raises
    for _ in range(5):
        _progress(clock, stage_id=1, segment_finished=False)
    assert sent == [] and len(logged) == 1 and "unit_output_complete" in logged[0]

    now[0] = DEFAULT_UNIT_TIMEOUT_S
    assert clock.expire() is True
    assert _decisions(sent) == [[("timed_out", "no_progress")]]


def test_a_raising_decision_hook_counts_as_no_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.engine.duplex.contracts import DuplexOutputAction, DuplexOutputDecision
    from vllm_omni.engine.duplex.session import input_clock as input_clock_module

    monkeypatch.setattr(input_clock_module.logger, "exception", lambda *a, **k: None)

    class _Raising(_DefaultHooksPlugin):
        def unit_decision(self, **kwargs):
            raise RuntimeError("plugin bug")

    clock, _ = _clock(_plugin(_Raising))
    decision = DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE, metadata={"model_listen": True})
    assert clock.unit_decision(stage_id=0, decision=decision) is None
    assert _clock()[0].unit_decision(stage_id=0, decision=decision) == DuplexUnitDecision(label="listen")


def test_timeout_warnings_are_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.engine.duplex.session import input_clock as input_clock_module

    warnings: list[str] = []
    monkeypatch.setattr(input_clock_module.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setattr(input_clock_module.logger, "debug", lambda *a, **k: None)
    now = [0.0]
    clock, sent = _clock(now=now)
    clock.unit_max_age_s = 1e9  # only the liveness valve, one unit per window
    for _ in range(5):
        _append(clock)
    for step in range(1, 6):
        now[0] = step * DEFAULT_UNIT_TIMEOUT_S
        assert clock.expire() is True
    assert len(sent) == 5
    assert len(warnings) == 3, warnings  # at 15 s, 45 s and 75 s: one per 30 s window
    assert "logged at debug level" in warnings[1]


def test_units_of_later_inputs_wait_for_their_own_acknowledgement() -> None:
    """Several units settled by one timeout check: each goes to the acknowledgement of the input that created it."""
    now = [0.0]
    clock, sent = _clock(now=now)
    clock.unit_timeout_s = 1e9  # only the maximum age
    for _ in range(2):
        _append(clock)
        _progress(clock, stage_id=0)  # both speak, neither finishes
    _append(clock, ms=200, units=0)
    now[0] = DEFAULT_UNIT_MAX_AGE_S
    assert clock.expire() is True

    assert [(e.input_index, e.audio_end_ms, e.unit_end_ms, len(e.units)) for e in sent] == [
        (1, 1000, 1000, 1),
        (2, 2000, 2000, 1),
        (3, 2200, 2000, 0),
    ]
    assert all(e.unit_end_ms <= e.audio_end_ms for e in sent)


def test_a_speaker_settled_for_its_maximum_age_keeps_its_place_when_another_unit_times_out() -> None:
    now = [0.0]
    clock, sent = _clock(now=now)
    (a,) = _append(clock)
    _progress(clock, stage_id=0)  # A speaks
    now[0] = DEFAULT_UNIT_MAX_AGE_S - 1
    clock.note_progress()
    (b,) = _append(clock)
    _progress(clock, stage_id=0)  # B speaks
    now[0] = DEFAULT_UNIT_MAX_AGE_S
    clock.note_progress()
    assert clock.expire() and a.complete  # A: max_age
    now[0] += DEFAULT_UNIT_TIMEOUT_S
    assert clock.expire() and b.complete  # B: no_progress

    _progress(clock, stage_id=1)  # A's final segment end, late
    _progress(clock, stage_id=1)  # B's

    assert list(clock._speaking) == [], "each late segment end went to its own unit"


def test_a_speaker_settled_by_the_liveness_valve_keeps_its_late_output_from_the_next_units() -> None:
    """Slow but correct: no unit is acknowledged before its own output after a transient stall."""
    now = [0.0]
    clock, sent = _clock(now=now)
    (a,) = _append(clock)
    _progress(clock, stage_id=0)  # A speaks
    now[0] += DEFAULT_UNIT_TIMEOUT_S  # nothing at all for 15 s
    assert clock.expire() and a.complete and clock._speaking[0] is a, "settled, still waiting for its output"
    (b,) = _append(clock)
    _progress(clock, stage_id=0)  # B speaks
    (c,) = _append(clock)
    _progress(clock, stage_id=0)  # C speaks

    _progress(clock, stage_id=1)  # A's final segment end, late: its own
    assert not b.complete
    _progress(clock, stage_id=1)  # B's
    assert b.complete and not c.complete, "C waits for its own output"
    _progress(clock, stage_id=1)  # C's
    assert [e.input_index for e in sent] == [1, 2, 3] and _decisions(sent)[1:] == [[("speak", None)]] * 2


def test_later_stage_progress_that_overtakes_its_stage0_segment_end_is_held_for_its_unit() -> None:
    """The orchestrator may deliver the Talker segment end and the final output before Stage 0's segment end."""

    class _NeedsUpstream(_DefaultHooksPlugin):
        def unit_output_complete(self, *, unit, final_stage_id, new_output, new_context, state, runtime_config):
            del final_stage_id, new_output, new_context, state, runtime_config
            return unit is not None and unit.upstream_segment_finished and unit.final_segment_finished

    clock, sent = _clock(_plugin(_NeedsUpstream))
    _append(clock)
    _progress(clock, stage_id=1, final_stage_id=2)  # Talker segment end, ahead of Stage 0
    _progress(clock, stage_id=2, final_stage_id=2)  # final segment end, ahead too
    assert sent == []

    _progress(clock, stage_id=0, final_stage_id=2)  # the unit's Stage-0 segment end: it speaks

    assert _decisions(sent) == [[("speak", None)]]
    (second,) = _append(clock)
    _progress(clock, stage_id=0, final_stage_id=2)
    assert not second.complete, "nothing held is left over for the next unit"


def test_later_stage_progress_with_no_undecided_stage0_submission_is_dropped() -> None:
    """A stray final segment end cannot be the next unit's: that unit was not even submitted yet."""
    clock, sent = _clock()
    (a,) = _append(clock)
    _progress(clock, stage_id=0)
    _progress(clock, stage_id=1)
    assert a.complete

    _progress(clock, stage_id=1)  # stray: no Stage-0 submission is undecided
    (b,) = _append(clock)
    _progress(clock, stage_id=0)  # B speaks

    assert not b.complete and not clock._held
    _progress(clock, stage_id=1)
    assert b.complete


def test_held_progress_is_dropped_when_its_epoch_has_no_undecided_submission_left() -> None:
    """Held for A, which then ends at Stage 0 without speaking: it is nobody's, not the next speaker's."""
    clock, sent = _clock()
    (a,) = _append(clock)
    _progress(clock, stage_id=1)  # held: A's Stage-0 submission is undecided
    assert len(clock._held) == 1
    _progress(clock, stage_id=0, decision="listen", ends_unit=True)  # A ends at Stage 0

    assert a.complete and not clock._held
    (b,) = _append(clock)
    _progress(clock, stage_id=0)
    assert not b.complete


def test_a_hold_overflow_credits_none_of_the_held_progress() -> None:
    """Past the bound, which held item belongs to which unit is unknown: all of it is dropped."""
    clock, sent = _clock()
    (a,) = _append(clock)
    for _ in range(300):
        _progress(clock, stage_id=1)
    assert not clock._held, "the overflow cleared the hold, and the epoch holds nothing more"

    _progress(clock, stage_id=0)  # A speaks: nothing is replayed
    assert not a.complete
    (b,) = _append(clock)
    _progress(clock, stage_id=0)
    assert not b.complete

    clock.cancel_open(before_epoch=1, reason="cancelled")  # a new epoch holds again
    _append(clock, epoch=1)
    _progress(clock, stage_id=1, epoch=1)
    assert len(clock._held) == 1


# --------------------------------------------------------------------------- #
# Audio accounting                                                            #
# --------------------------------------------------------------------------- #


def test_audio_end_and_unit_end_are_exact_with_odd_sized_appends() -> None:
    """3000 appends of one 48 kHz sample are 62.5 ms, not 3000 x 20 us = 60 ms."""
    from fractions import Fraction

    clock, sent = _clock()
    for _ in range(3000):
        ack = clock.begin_input(APPEND)
        clock.note_input_audio(ack, samples=1, sample_rate_hz=48000)
        clock.end_input(ack)
    unit = clock.new_unit(input_us=Fraction(3000 * 1_000_000, 48000), epoch=0)
    clock.settle_unit(unit, "dropped")
    _append(clock, ms=0, units=0)

    assert (sent[-1].audio_end_ms, sent[-1].unit_end_ms) == (62, 62)
