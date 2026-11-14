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
        # First continuation anchors to the last real submission: unit N was
        # submitted at t=1.0; the first deadline allows 5 ms for input arrival.
        (1.0, 1.4, 1.0, None, 0.605, 3.005),
        # Only the remaining budget is slept (audio produced at 1.7).
        (1.0, 1.7, 1.0, None, 0.305, 3.005),
        # Overdue deadline: no sleep; the chain still advances by the period.
        (1.0, 2.5, 1.0, None, 0.0, 3.005),
        # No submission yet (None): the first continuation anchors to now.
        (1.0, 0.4, None, None, 1.0, 2.4),
        # Deadlines advance from the current deadline, not from now, so pipeline
        # processing time does not accumulate as timer drift.
        (1.0, 4.2, 3.0, 5.0, 0.8, 6.0),
        # Consecutive late units keep a 1 s cadence without repeatedly adding the grace.
        (1.0, 2.2, 1.0, None, 0.0, 3.005),
        (1.0, 3.2, 2.2, 3.005, 0.0, 4.005),
        # A short overshoot (within one period of the current deadline) still
        # chases the stale deadline: immediate submit, chain advances.
        (1.0, 2.5, 1.0, 2.0, 0.0, 3.0),
        # A long stall (more than one period past the current deadline) submits
        # one continuation immediately and restarts from that submission.
        (1.0, 5.0, 1.0, 2.0, 0.0, 6.0),
        # The same recovery applies to the first continuation after a stall.
        (1.0, 5.0, 1.0, None, 0.0, 6.0),
        # Zero chunk period (guarded upstream): total, no sleep.
        (0.0, 10.0, None, None, 0.0, 10.0),
        # The initial grace does not accumulate over consecutive 80 ms frames.
        (0.08, 1.08, 1.0, None, 0.005, 1.165),
        (0.08, 1.17, 1.085, 1.165, 0.0, 1.245),
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
