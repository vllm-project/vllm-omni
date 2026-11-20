# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deterministic unit tests for deadline-aligned silence continuation.

The engine-resident duplex scheduler in
``vllm_omni/engine/duplex/session/runner.py`` aligns each silence
continuation to ``submission_time_N + chunk_period`` and sleeps only the
remaining budget. These tests cover the pure deadline arithmetic and the
per-session reset semantics of the shared ``DuplexModelSessionState``
contract.
"""

from __future__ import annotations

import pytest

from vllm_omni.engine.duplex.session.runner import (
    compute_silence_continuation_deadline,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import (
    MiniCPMO45ServingSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("chunk_period_s", "now", "last_submit", "current_deadline", "delay_s", "next_silence_deadline"),
    [
        # First continuation anchors to the last real submission plus the
        # jitter margin (min(2 * chunk_period_s, 0.25) = 0.25 here): unit N
        # was submitted at t=1.0, audio N produced at 1.4 -> 0.85 s of sleep
        # left (deadline 2.25, not 2.0).
        (1.0, 1.4, 1.0, None, 0.85, 3.25),
        # Only the remaining budget is slept (audio produced at 1.7).
        (1.0, 1.7, 1.0, None, 0.55, 3.25),
        # Overdue deadline: no sleep; the chain still advances by the period.
        (1.0, 2.5, 1.0, None, 0.0, 3.25),
        # No submission yet (None): the first continuation anchors to now,
        # plus the jitter margin.
        (1.0, 0.4, None, None, 1.25, 2.65),
        # Deadlines advance from the current deadline, not from now, so pipeline
        # processing time does not accumulate as timer drift. The margin is
        # only added when anchoring a fresh chain (current_deadline is None).
        (1.0, 4.2, 3.0, 5.0, 0.8, 6.0),
        # A unit well short of a fresh chain's margined deadline still sleeps
        # the remaining budget.
        (1.0, 2.2, 1.0, None, 0.05, 3.25),
        (1.0, 3.2, 2.0, 3.0, 0.0, 4.0),
        # A short overshoot (within one period of the current deadline) still
        # chases the stale deadline: immediate submit, chain advances.
        (1.0, 2.5, 1.0, 2.0, 0.0, 3.0),
        # A long stall (more than one period past the current deadline) submits
        # one continuation immediately and restarts from that submission.
        (1.0, 5.0, 1.0, 2.0, 0.0, 6.0),
        # The same recovery applies to the first continuation after a stall;
        # the margin added to the stale base deadline is absorbed by the
        # staleness restart, so the outcome matches the un-margined case.
        (1.0, 5.0, 1.0, None, 0.0, 6.0),
        # Zero chunk period (guarded upstream) does not get a jitter margin:
        # there is no periodic cadence to protect, so this stays a no-op.
        (0.0, 10.0, None, None, 0.0, 10.0),
    ],
)
def test_compute_silence_continuation_deadline(
    chunk_period_s: float,
    now: float,
    last_submit: float | None,
    current_deadline: float | None,
    delay_s: float,
    next_silence_deadline: float,
) -> None:
    delay, next_dl = compute_silence_continuation_deadline(
        chunk_period_s=chunk_period_s,
        now=now,
        last_submit=last_submit,
        current_deadline=current_deadline,
    )
    assert delay == pytest.approx(delay_s)
    assert next_dl == pytest.approx(next_silence_deadline)


def test_continuation_deadline_does_not_fire_at_exactly_one_chunk_period() -> None:
    """A client streaming real input at exactly chunk_period_s must not race the continuation.

    Without a jitter margin, a continuation anchored at ``last_submit`` with
    ``now == last_submit + chunk_period_s`` returns ``delay_s == 0`` and fires
    immediately. A real client paced at the same period then submits its next
    append microseconds later, landing as a second, separate model step
    instead of superseding the now-redundant continuation -- on a sustained
    stream this double-steps the model on every cycle that loses the race,
    producing audio faster than real time (observed as unbounded
    ``arrival_backlog`` growth in long-running sessions).
    """
    chunk_period_s = 0.08  # Nemotron VoiceChat's chunk_period_ms.
    last_submit = 100.0
    delay_s, _ = compute_silence_continuation_deadline(
        chunk_period_s=chunk_period_s,
        now=last_submit + chunk_period_s,
        last_submit=last_submit,
        current_deadline=None,
    )
    assert delay_s > 0.0


def test_continuation_deadline_survives_realistic_client_jitter() -> None:
    """A fixed few-ms margin is not enough: real client jitter (network,
    scheduling, bursty delivery) regularly exceeds it. A live H100 duplex
    session streaming real-time-paced input with a naive 30 ms margin still
    lost the race on 12% of appends (269/2250 over a 3-minute run), with
    observed capture-interval jitter up to p99=107ms and max=206ms against
    an 80 ms nominal period. The margin must comfortably exceed that tail.
    """
    chunk_period_s = 0.08  # Nemotron VoiceChat's chunk_period_ms.
    last_submit = 100.0
    # 150ms late is past the old 30ms margin and the observed p99/max jitter.
    delay_s, _ = compute_silence_continuation_deadline(
        chunk_period_s=chunk_period_s,
        now=last_submit + chunk_period_s + 0.15,
        last_submit=last_submit,
        current_deadline=None,
    )
    assert delay_s > 0.0


def test_continuation_jitter_margin_caps_for_large_chunk_periods() -> None:
    """The margin scales with chunk_period_s but is capped, so a model with a
    large native unit (e.g. a 1s chunk_period_s) doesn't get an excessive
    multi-second delay before idle-continuation kicks in.
    """
    delay_s, _ = compute_silence_continuation_deadline(
        chunk_period_s=1.0,
        now=100.0 + 1.0 + 0.25,
        last_submit=100.0,
        current_deadline=None,
    )
    assert delay_s == pytest.approx(0.0)
    delay_s, _ = compute_silence_continuation_deadline(
        chunk_period_s=1.0,
        now=100.0 + 1.0 + 0.24,
        last_submit=100.0,
        current_deadline=None,
    )
    assert delay_s == pytest.approx(0.01)


class TestSilenceDeadlineSessionState:
    def test_clear_continuation_resets_the_deadline_chain(self) -> None:
        # The native-duplex session state feeds the shared runner, so it must
        # start with an unset chain and drop it at turn boundaries.
        state = MiniCPMO45ServingSessionState()
        assert state.last_native_submit_monotonic is None
        assert state.silence_deadline_monotonic is None
        state.last_native_submit_monotonic = 5.0
        state.silence_deadline_monotonic = 6.0
        state.clear_continuation()
        assert state.last_native_submit_monotonic is None
        assert state.silence_deadline_monotonic is None
