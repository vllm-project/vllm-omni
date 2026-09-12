# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deterministic unit tests for deadline-aligned silence continuation.

The native-duplex scheduler in
``vllm_omni/entrypoints/duplex/session_runner.py`` aligns each silence
continuation to ``submission_time_N + chunk_period`` and sleeps only the
remaining budget. These tests cover the pure deadline arithmetic and the
per-session reset semantics shared by every native-duplex serving adapter.
"""

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.duplex.session_runner import (
    compute_silence_continuation_deadline,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import (
    MiniCPMO45ServingSessionState,
)
from vllm_omni.model_executor.models.personaplex.duplex.serving_adapter import (
    PersonaPlexServingSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("chunk_period_s", "now", "last_submit", "current_deadline", "delay_s", "next_silence_deadline"),
    [
        # First continuation anchors to the last real submission: unit N was
        # submitted at t=1.0, audio N produced at 1.4 -> 0.6 s of sleep left.
        (1.0, 1.4, 1.0, None, 0.6, 3.0),
        # Only the remaining budget is slept (audio produced at 1.7).
        (1.0, 1.7, 1.0, None, 0.3, 3.0),
        # Overdue deadline: no sleep; the chain still advances by the period.
        (1.0, 2.5, 1.0, None, 0.0, 3.0),
        # No submission yet (None): the first continuation anchors to now.
        (1.0, 0.4, None, None, 1.0, 2.4),
        # Deadlines advance from the current deadline, not from now, so pipeline
        # processing time does not accumulate as timer drift.
        (1.0, 4.2, 3.0, 5.0, 0.8, 6.0),
        # Two consecutive units that overrun their deadline by 0.2 s keep the
        # 1 s cadence: no sleep, and the chain advances one period each.
        (1.0, 2.2, 1.0, None, 0.0, 3.0),
        (1.0, 3.2, 2.0, 3.0, 0.0, 4.0),
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
    @pytest.mark.parametrize(
        "state",
        [
            MiniCPMO45ServingSessionState(),
            PersonaPlexServingSessionState(),
        ],
    )
    def test_clear_continuation_resets_the_deadline_chain(self, state: object) -> None:
        # Both native-duplex session states feed the shared runner, so both
        # must start with an unset chain and drop it at turn boundaries.
        assert state.last_native_submit_monotonic is None
        assert state.silence_deadline_monotonic is None
        state.last_native_submit_monotonic = 5.0
        state.silence_deadline_monotonic = 6.0
        state.clear_continuation()
        assert state.last_native_submit_monotonic is None
        assert state.silence_deadline_monotonic is None
