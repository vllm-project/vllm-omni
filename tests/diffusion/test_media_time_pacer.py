# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the media-time release schedule.

Every case drives an injected clock, so nothing here sleeps and nothing depends
on how fast the test machine is.
"""

import pytest

from vllm_omni.diffusion.media_time_pacer import MediaTimePacer

FPS = 16.0
CHUNK = 12  # frames, i.e. 0.75 s of media
CHUNK_SECONDS = CHUNK / FPS


class FakeClock:
    """Monotonic clock the test moves by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_rejects_a_nonsense_configuration():
    with pytest.raises(ValueError):
        MediaTimePacer(0)
    with pytest.raises(ValueError):
        MediaTimePacer(FPS, lead_seconds=-1.0)
    with pytest.raises(ValueError):
        MediaTimePacer(FPS, max_lag_seconds=-1.0)
    with pytest.raises(ValueError):
        MediaTimePacer(FPS).delay_before_release(-1)


def test_the_first_chunk_is_never_held():
    # The viewer's clock starts when the viewer gets something to watch. Holding
    # the opening chunk would add latency to the one place a user can see it.
    clock = FakeClock()
    pacer = MediaTimePacer(FPS, clock=clock)
    assert pacer.delay_before_release(CHUNK) == 0.0


def test_generation_faster_than_real_time_is_held_to_the_schedule():
    clock = FakeClock()
    pacer = MediaTimePacer(FPS, clock=clock)
    pacer.delay_before_release(CHUNK)

    # Produced in a quarter of the time it takes to watch.
    clock.advance(CHUNK_SECONDS / 4)
    delay = pacer.delay_before_release(CHUNK)
    assert delay == pytest.approx(CHUNK_SECONDS * 3 / 4)

    # Honouring the delay puts the next deadline exactly one chunk later.
    clock.advance(delay + CHUNK_SECONDS / 4)
    assert pacer.delay_before_release(CHUNK) == pytest.approx(CHUNK_SECONDS * 3 / 4)


def test_generation_slower_than_real_time_is_never_delayed():
    # A session already missing its deadlines must not be slowed further; the
    # pacer has to cost nothing in the case that needs the time most.
    clock = FakeClock()
    pacer = MediaTimePacer(FPS, clock=clock)
    pacer.delay_before_release(CHUNK)
    for _ in range(10):
        clock.advance(CHUNK_SECONDS * 3)
        assert pacer.delay_before_release(CHUNK) == 0.0


def test_lead_releases_early_by_exactly_the_lead():
    clock = FakeClock()
    lead = 0.25
    pacer = MediaTimePacer(FPS, lead_seconds=lead, clock=clock)
    pacer.delay_before_release(CHUNK)
    assert pacer.delay_before_release(CHUNK) == pytest.approx(CHUNK_SECONDS - lead)


def test_the_schedule_does_not_drift_over_a_long_session():
    # Deadlines are absolute against one origin rather than accumulated per
    # chunk, so per-chunk rounding cannot pile up.
    clock = FakeClock()
    pacer = MediaTimePacer(FPS, clock=clock)
    origin = clock.now
    pacer.delay_before_release(CHUNK)
    for index in range(1, 400):
        clock.advance(0.01)
        delay = pacer.delay_before_release(CHUNK)
        clock.advance(delay)
        assert clock.now == pytest.approx(origin + index * CHUNK_SECONDS)
    assert pacer.released_frames == 400 * CHUNK


def test_a_stall_is_written_off_rather_than_repaid_as_a_burst():
    # The behaviour this guard exists for: after a long stall a fixed origin
    # leaves every later deadline in the past, so the session would dump its
    # backlog at full speed -- exactly the run-ahead the pacer prevents.
    clock = FakeClock()
    pacer = MediaTimePacer(FPS, max_lag_seconds=1.0, clock=clock)
    pacer.delay_before_release(CHUNK)

    clock.advance(30.0)  # a stall far beyond the allowance
    assert pacer.delay_before_release(CHUNK) == 0.0
    assert pacer.rebases == 1

    # Back on schedule immediately, not after catching up 30 s of debt.
    clock.advance(CHUNK_SECONDS / 4)
    assert pacer.delay_before_release(CHUNK) == pytest.approx(CHUNK_SECONDS * 3 / 4)


def test_without_a_lag_allowance_the_debt_is_kept():
    # The opposite policy, for a caller that wants catch-up: the origin never
    # moves, so the session runs flat out until it is level with the schedule.
    clock = FakeClock()
    pacer = MediaTimePacer(FPS, max_lag_seconds=None, clock=clock)
    pacer.delay_before_release(CHUNK)

    clock.advance(30.0)
    for _ in range(20):
        assert pacer.delay_before_release(CHUNK) == 0.0
    assert pacer.rebases == 0


def test_accounting_is_independent_of_whether_the_caller_waits():
    # A caller that drops a chunk or ignores the delay still gets a correct
    # schedule afterwards, because frames are counted at call time.
    clock = FakeClock()
    honoured = MediaTimePacer(FPS, clock=clock)
    honoured.delay_before_release(CHUNK)
    honoured.delay_before_release(CHUNK)
    assert honoured.released_frames == 2 * CHUNK
    assert honoured.released_media_seconds == pytest.approx(2 * CHUNK_SECONDS)


def test_variable_chunk_sizes_follow_the_frames_not_the_chunks():
    # The opening chunk of a causal video model is shorter than the rest, so a
    # per-chunk schedule would be wrong from the first tick onwards.
    clock = FakeClock()
    pacer = MediaTimePacer(FPS, clock=clock)
    pacer.delay_before_release(9)  # opening chunk
    assert pacer.delay_before_release(CHUNK) == pytest.approx(9 / FPS)
    assert pacer.released_frames == 9 + CHUNK
