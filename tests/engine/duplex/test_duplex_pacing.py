# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``duplex_session.pacing``: the playback mirror, the session pacing state machine and its lead."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from vllm_omni.config.stage_config import DuplexPacingConfig, DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.session.pacing import (
    QUIET_GUARD_MIN_S,
    PaceState,
    PlaybackMirror,
    SessionPacing,
    same_turn_response_keeps_continuation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

#: Past the quiet guard of a commit at 0.0 with no appends before it.
QUIET = QUIET_GUARD_MIN_S


def _reference_playback(chunks: list[tuple[float, float]], buffer_s: float) -> tuple[float, float]:
    """duplex_rt / rtf_load ``playback``: (stall, play end) from the first arrival plus ``buffer_s``."""
    play_end = chunks[0][0] + buffer_s + chunks[0][1]
    stall = 0.0
    for arrival, duration in chunks[1:]:
        if arrival > play_end:
            stall += arrival - play_end
            play_end = arrival
        play_end += duration
    return stall, play_end


def _pacing(**pacing: object) -> SessionPacing:
    defaults: dict[str, object] = {"enabled": True, "onset_lead_max_s": 0.9}
    defaults.update(pacing)
    session = SimpleNamespace(active_response_id=None)
    return SessionPacing(DuplexSessionRuntimeConfig(pacing=defaults), session)


def _emit(pace: SessionPacing, now: float, seconds: float, response_id: str = "resp_1") -> None:
    """One audio delta of ``response_id`` (the session's active response while it plays)."""
    pace.session.active_response_id = response_id
    pace.on_audio_emit(now, seconds)


@pytest.mark.parametrize("seed", range(20))
def test_mirror_matches_the_duplex_rt_playback_rule(seed: int) -> None:
    rng = random.Random(seed)
    mirror = PlaybackMirror(prebuffer_s=0.5)
    t = 10.0
    chunks: list[tuple[float, float]] = []
    stall = 0.0
    for _ in range(rng.randint(1, 30)):
        t += rng.uniform(0.0, 1.6)
        duration = rng.choice([0.12, 0.5, 1.0, 1.12])
        chunks.append((t, duration))
        stall += mirror.on_emit(t, duration)
    want_stall, want_end = _reference_playback(chunks, 0.5)
    assert stall == pytest.approx(want_stall)
    assert mirror.play_end == pytest.approx(want_end)


def test_mirror_buffers_once_per_session_and_cuts() -> None:
    mirror = PlaybackMirror()
    assert mirror.prebuffer_s == 0.5
    assert mirror.unplayed(0.0) == 0.0
    mirror.on_emit(1.0, 0.12)
    assert mirror.play_end == pytest.approx(1.62)
    assert mirror.unplayed(1.0) == pytest.approx(0.62)
    mirror.cut(1.2)
    assert mirror.play_end == pytest.approx(1.2)
    assert mirror.response_first_audio_t is None
    # A later response's first chunk re-anchors without a prebuffer and is not a stall.
    assert mirror.on_emit(5.0, 1.0) == 0.0
    assert mirror.play_end == pytest.approx(6.0)
    # Inside a response a late chunk is a stall.
    assert mirror.on_emit(6.5, 1.0) == pytest.approx(0.5)
    assert mirror.response_stall_s == pytest.approx(0.5)


@pytest.mark.parametrize(("d1", "lead"), [(0.04, 0.9), (0.12, 0.9), (0.84, 0.26), (1.0, 0.1), (1.12, 0.0)])
def test_lead_follows_the_first_chunk(d1: float, lead: float) -> None:
    pace = _pacing()
    pace.on_commit(0.0)
    assert pace.state == PaceState.ONSET
    # d1 unknown: the full lead, so unit 2 fires as soon as unit 1's segment ends.
    assert pace.lead_s(QUIET) == pytest.approx(0.9)
    _emit(pace, 0.4, d1)
    assert pace.state == PaceState.PACED
    assert pace.lead_s(0.5) == pytest.approx(lead)


def test_state_machine_critical_and_recovery() -> None:
    pace = _pacing()
    pace.on_commit(0.0)
    _emit(pace, 0.3, 0.84)  # play end 0.3 + 0.5 + 0.84 = 1.64
    assert pace.lead_s(0.5) == pytest.approx(0.26)
    _emit(pace, 1.75, 1.0)  # 0.11 s stall
    assert pace.state == PaceState.CRITICAL
    assert pace.lead_s(1.8) == pytest.approx(0.9)
    _emit(pace, 2.0, 1.0)
    assert pace.state == PaceState.CRITICAL
    _emit(pace, 2.5, 1.0)
    assert pace.state == PaceState.PACED


def test_the_quiet_guard_holds_the_lead_back_right_after_the_commit() -> None:
    pace = _pacing(onset_skip_response_wait=True)
    pace.on_commit(10.0)
    assert pace.lead_s(10.1) == 0.0
    assert pace.skip_response_wait(10.1) is False
    assert pace.recheck_in(10.2) == pytest.approx(QUIET_GUARD_MIN_S - 0.2)  # wakes when the guard passes
    assert pace.lead_s(10.0 + QUIET) == pytest.approx(0.9)
    assert pace.skip_response_wait(10.0 + QUIET) is True


def test_a_long_chunk_client_with_its_mic_open_is_not_taken_for_quiet() -> None:
    """Client 2 pushing 600 ms chunks: its next append after the commit lands inside the guard."""
    pace = _pacing(onset_skip_response_wait=True)
    for index in range(5):
        pace.on_client_append(10.0 + 0.6 * index, is_speech=True, duration_s=0.6)
    pace.on_commit(12.5)
    assert pace.lead_s(12.8) == 0.0  # guard 1.5 x 0.6 = 0.9 s
    assert pace.skip_response_wait(12.8) is False
    assert pace.lead_s(13.45) == pytest.approx(0.9)  # a client that really went quiet still leads
    # The pause before the next utterance is no chunk interval: that commit uses the floor.
    pace.on_client_append(20.0, is_speech=True, duration_s=0.2)
    pace.on_client_append(20.2, is_speech=True, duration_s=0.2)
    pace.on_commit(20.4)
    assert pace.lead_s(20.4 + 0.35) == pytest.approx(0.9)  # guard 1.5 x 0.2 = 0.3 s


def test_any_input_after_commit_returns_the_reply_to_the_wall_clock() -> None:
    pace = _pacing(onset_skip_response_wait=True)
    pace.on_commit(0.0)
    assert pace.skip_response_wait(QUIET) is True
    pace.on_client_append(0.3, is_speech=False, duration_s=0.2)
    assert pace.lead_s(0.4) == 0.0
    assert pace.skip_response_wait(0.4) is False
    _emit(pace, 0.4, 0.12)
    assert pace.lead_s(0.5) == 0.0
    # The next commit makes the client quiet again.
    pace.session.active_response_id = None
    pace.on_commit(5.0)
    assert pace.lead_s(5.0 + QUIET) == pytest.approx(0.9)


def test_a_reply_ended_by_any_path_resets_the_next_turn_to_onset() -> None:
    """Turn 1 ends with a listen (no end_of_turn hook): turn 2's commit still starts at ONSET."""
    pace = _pacing(onset_skip_response_wait=True)
    pace.on_commit(0.0)
    _emit(pace, 0.4, 0.5)
    assert pace.state == PaceState.PACED
    pace.session.active_response_id = None  # ended: completed, listen, drained ...
    assert pace.lead_s(3.0) == 0.0
    assert pace.state == PaceState.LISTEN
    pace.on_commit(5.0)
    assert pace.state == PaceState.ONSET and pace.d1 is None
    assert pace.skip_response_wait(5.0 + QUIET) is True
    assert pace.lead_s(5.0 + QUIET) == pytest.approx(0.9)


def test_a_draining_replys_tail_plays_but_is_not_the_new_replys_first_chunk() -> None:
    pace = _pacing()
    pace.on_commit(0.0)
    pace.session.active_response_id = "resp_2"
    pace.on_audio_emit(0.3, 1.0, draining=True)  # resp_1's tail, still draining
    assert pace.d1 is None and pace.state == PaceState.ONSET
    assert pace.mirror.play_end == pytest.approx(0.3 + 0.5 + 1.0)
    pace.on_audio_emit(0.4, 0.12)
    assert pace.d1 == pytest.approx(0.12) and pace.state == PaceState.PACED
    assert pace.mirror.play_end == pytest.approx(1.92)


def test_a_reply_that_spoke_before_the_commit_stays_paced() -> None:
    pace = _pacing()
    _emit(pace, 0.4, 0.84)  # Stage-0 handoff before the client's commit
    pace.on_commit(1.0)
    assert pace.state == PaceState.PACED
    assert pace.lead_s(1.0 + QUIET) == pytest.approx(0.26)


def test_skip_response_wait_needs_onset_and_its_switch() -> None:
    assert _pacing().skip_response_wait(QUIET) is False
    pace = _pacing(onset_skip_response_wait=True)
    assert pace.skip_response_wait(QUIET) is False  # no commit yet
    pace.on_commit(0.0)
    assert pace.skip_response_wait(QUIET) is True
    _emit(pace, 0.3, 0.5)
    assert pace.skip_response_wait(0.4) is False  # PACED


def test_a_quiet_client_keeps_its_reply_on_a_model_listen() -> None:
    pace = _pacing()
    assert pace.keeps_reply_on_listen() is False  # no commit: a listen may be a barge-in
    pace.on_commit(0.0)
    assert pace.keeps_reply_on_listen() is True  # also inside the guard: no input since the commit
    pace.on_client_append(0.5, is_speech=True, duration_s=0.2)
    assert pace.keeps_reply_on_listen() is False
    off = _pacing(enabled=False)
    off.on_commit(0.0)
    assert off.keeps_reply_on_listen() is False


def test_epoch_advance_resets_to_listen_and_cuts_the_mirror() -> None:
    pace = _pacing()
    pace.on_commit(0.0)
    _emit(pace, 1.0, 1.0)
    pace.session.active_response_id = None
    pace.on_epoch_advance(1.2)
    assert pace.state == PaceState.LISTEN
    assert pace.lead_s(1.3) == 0.0
    assert pace.mirror.unplayed(1.2) == 0.0


def test_lead_is_zero_when_disabled_or_unset() -> None:
    off = _pacing(enabled=False)
    off.on_commit(0.0)
    assert off.lead_s(QUIET) == 0.0
    no_lead = _pacing(onset_lead_max_s=0.0)
    no_lead.on_commit(0.0)
    assert no_lead.lead_s(QUIET) == 0.0
    assert no_lead.speech_fire_at(10.0, 9.0) == 10.0
    assert no_lead.idle_fire_at(10.0, 9.0) == 10.0


def test_speech_fire_at_subtracts_the_lead() -> None:
    pace = _pacing()
    pace.on_commit(0.0)
    assert pace.speech_fire_at(1.0, QUIET) == pytest.approx(0.1)


def test_same_turn_keep_needs_the_skip_switch() -> None:
    assert same_turn_response_keeps_continuation(None) is False
    assert same_turn_response_keeps_continuation(_pacing()) is False
    assert same_turn_response_keeps_continuation(_pacing(onset_skip_response_wait=True)) is True


def test_pacing_config_defaults_are_off() -> None:
    config = DuplexPacingConfig()
    assert config.enabled is False
    assert config.onset_lead_max_s == 0.0
    assert config.onset_skip_response_wait is False
    assert config.fire_grid_ms == 0
    assert config.idle_grid is False
    assert DuplexSessionRuntimeConfig().pacing == config


def _paced_grid(*, lead_max: float = 0.9, d1: float = 0.84, **pacing: object) -> SessionPacing:
    pace = _pacing(onset_lead_max_s=lead_max, fire_grid_ms=250, **pacing)
    pace.on_commit(100.0)
    _emit(pace, 100.3, d1)
    assert pace.state == PaceState.PACED
    return pace


def test_snap_is_the_identity_without_a_grid() -> None:
    pace = _pacing()
    pace.on_commit(100.0)
    _emit(pace, 100.3, 0.84)
    assert pace.snap(101.74, 102.0, 100.6) == 101.74
    assert pace.snap_holds(100.6) is False


def test_snap_only_moves_earlier_within_the_lead_budget() -> None:
    pace = _paced_grid()
    assert pace.snap(101.74, 102.0, 100.6) == pytest.approx(101.5)
    assert pace.snap_holds(100.6) is True
    # The earlier grid point is already past: unchanged, never rounded up.
    assert pace.snap(101.74, 102.0, 101.6) == pytest.approx(101.74)
    # 102.03 -> 102.0 would lead 0.13 s > 0.1: unchanged.
    tight = _paced_grid(lead_max=0.1, d1=1.0)
    assert tight.snap(102.03, 102.13, 100.6) == pytest.approx(102.03)


def test_eager_states_never_snap() -> None:
    pace = _pacing(fire_grid_ms=250)
    pace.on_commit(100.0)
    assert pace.state == PaceState.ONSET
    assert pace.snap(101.74, 102.0, 100.6) == 101.74
    critical = _paced_grid()
    _emit(critical, 102.5, 1.0)  # a stall
    assert critical.state == PaceState.CRITICAL
    assert critical.snap(101.74, 102.0, 100.6) == 101.74
    streaming = _paced_grid()
    streaming.on_client_append(100.7, is_speech=False, duration_s=0.2)
    assert streaming.snap(101.74, 102.0, 100.8) == 101.74
    assert streaming.snap_holds(100.8) is False


def test_idle_grid_rounds_up_by_at_most_one_grid_step() -> None:
    pace = _paced_grid(idle_grid=True)
    assert pace.idle_fire_at(101.1, 100.6) == pytest.approx(101.25)
    assert pace.idle_fire_at(101.25, 100.6) == pytest.approx(101.25)
    assert _paced_grid().idle_fire_at(101.1, 100.6) == 101.1  # idle_grid off
    pace.on_client_append(101.0, is_speech=True, duration_s=0.2)
    assert pace.idle_fire_at(101.1, 101.0) == 101.1  # not a quiet client any more
