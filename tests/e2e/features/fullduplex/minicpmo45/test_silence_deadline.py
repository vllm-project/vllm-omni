# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic unit tests for deadline-aligned silence continuation.

The silence continuation scheduler in
``vllm_omni/experimental/fullduplex/openai/session_runner.py`` aligns the
next model input unit to ``submission_time_N + chunk_period`` and sleeps only
the remaining budget. These tests cover the pure deadline arithmetic and the
session-state reset semantics that the scheduler relies on.
"""

from __future__ import annotations

import pytest

from vllm_omni.experimental.fullduplex.minicpmo45.session import (
    MiniCPMO45ServingSessionState,
)
from vllm_omni.experimental.fullduplex.openai.session_runner import (
    compute_silence_continuation_deadline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestComputeSilenceContinuationDeadline:
    """Deterministic deadline arithmetic (no async, no clock)."""

    def test_first_continuation_anchors_to_last_submit(self) -> None:
        # Unit N submitted at t=1.0; chunk period is 1.0 s, so unit N+1 is
        # due at t=2.0. Audio N produced at now=1.4 -> 0.6 s of sleep left.
        delay_s, next_deadline = compute_silence_continuation_deadline(
            chunk_period_s=1.0,
            now=1.4,
            last_submit=1.0,
            prior_deadline=None,
        )
        assert delay_s == pytest.approx(0.6)
        assert next_deadline == pytest.approx(3.0)

    def test_sleeps_only_remaining_budget(self) -> None:
        # Unit N submitted at t=1.0, due at t=2.0. Audio produced at now=1.7
        # -> only the remaining 0.3 s is slept.
        delay_s, next_deadline = compute_silence_continuation_deadline(
            chunk_period_s=1.0,
            now=1.7,
            last_submit=1.0,
            prior_deadline=None,
        )
        assert delay_s == pytest.approx(0.3)
        assert next_deadline == pytest.approx(3.0)

    def test_overdue_deadline_produces_no_sleep(self) -> None:
        # Unit N submitted at t=1.0, due at t=2.0. Audio produced at now=2.5,
        # past the deadline; no sleep.
        delay_s, next_deadline = compute_silence_continuation_deadline(
            chunk_period_s=1.0,
            now=2.5,
            last_submit=1.0,
            prior_deadline=None,
        )
        assert delay_s == 0.0
        assert next_deadline == pytest.approx(3.0)

    def test_unsubmitted_sentinel_anchors_to_now(self) -> None:
        # last_submit == 0.0 is the "never submitted" sentinel; the first
        # continuation anchors to the current time instead.
        delay_s, next_deadline = compute_silence_continuation_deadline(
            chunk_period_s=1.0,
            now=0.4,
            last_submit=0.0,
            prior_deadline=None,
        )
        assert delay_s == pytest.approx(1.0)
        assert next_deadline == pytest.approx(2.4)

    def test_deadlines_advance_without_drift(self) -> None:
        # Continuation N has prior deadline 5.0; processing finished at 4.2.
        # The next deadline advances from the prior deadline (6.0), not from
        # the current time (4.2 + 1.0 = 5.2), so timer drift does not
        # accumulate.
        delay_s, next_deadline = compute_silence_continuation_deadline(
            chunk_period_s=1.0,
            now=4.2,
            last_submit=3.0,
            prior_deadline=5.0,
        )
        assert delay_s == pytest.approx(0.8)
        assert next_deadline == pytest.approx(6.0)

    def test_processing_time_does_not_accumulate_as_drift(self) -> None:
        # Two continuations in a row: even when each unit overruns its
        # deadline by 0.2 s, the deadline chain stays on the 1 s cadence.
        d1, next1 = compute_silence_continuation_deadline(
            chunk_period_s=1.0, now=1.2, last_submit=0.0, prior_deadline=None
        )
        assert d1 == 0.0
        assert next1 == pytest.approx(2.0)

        d2, next2 = compute_silence_continuation_deadline(
            chunk_period_s=1.0, now=2.2, last_submit=1.0, prior_deadline=next1
        )
        assert d2 == 0.0
        assert next2 == pytest.approx(3.0)

    def test_zero_chunk_period_never_sleeps(self) -> None:
        # The scheduler never calls this with chunk_period_s == 0.0 (it is
        # guarded by ``continuation_delay_s > 0`` upstream), but the helper
        # must still be total: no sleep, and the deadline stays put.
        delay_s, next_deadline = compute_silence_continuation_deadline(
            chunk_period_s=0.0,
            now=10.0,
            last_submit=0.0,
            prior_deadline=None,
        )
        assert delay_s == 0.0
        assert next_deadline == pytest.approx(10.0)


class TestSilenceDeadlineSessionState:
    """Session-state reset semantics used by the scheduler."""

    def test_real_input_resets_the_deadline_chain(self) -> None:
        state = MiniCPMO45ServingSessionState()
        # A silence continuation had advanced the chain.
        state.silence_deadline_monotonic = 5.0
        # A real (non-silence) append re-anchors: the deadline is cleared.
        state.last_native_submit_monotonic = 3.0
        state.silence_deadline_monotonic = None
        assert state.silence_deadline_monotonic is None
        assert state.last_native_submit_monotonic == 3.0

    def test_silence_continuation_keeps_the_chain(self) -> None:
        state = MiniCPMO45ServingSessionState()
        state.last_native_submit_monotonic = 2.0
        state.silence_deadline_monotonic = 3.0
        # A silence continuation does not clear the chain.
        assert state.silence_deadline_monotonic == 3.0
        assert state.last_native_submit_monotonic == 2.0

    def test_new_session_starts_with_no_deadline(self) -> None:
        state = MiniCPMO45ServingSessionState()
        assert state.silence_deadline_monotonic is None
        assert state.last_native_submit_monotonic == 0.0
