# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the seeded-request admission policy.

Seeded TTS requests are only bit-reproducible when they never share an
autoregressive batch (see ``vllm_omni/core/sched/seeded_isolation.py`` for the
measurements). These tests pin the decision table only; the scheduler integration
is exercised by the GPU tests.
"""

import pytest

from vllm_omni.core.sched.seeded_isolation import isolation_plan, request_is_seeded

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Params:
    def __init__(self, seed):
        self.seed = seed


def test_idle_with_seeded_waiting_is_promoted() -> None:
    assert isolation_plan(has_running=False, seeded_running=False, seeded_waiting=True) == (False, True)


def test_seeded_running_defers_everyone() -> None:
    assert isolation_plan(has_running=True, seeded_running=True, seeded_waiting=False) == (True, False)


def test_seeded_waiting_waits_for_running_requests() -> None:
    assert isolation_plan(has_running=True, seeded_running=False, seeded_waiting=True) == (True, False)


def test_unseeded_traffic_is_untouched() -> None:
    assert isolation_plan(has_running=True, seeded_running=False, seeded_waiting=False) == (False, False)
    assert isolation_plan(has_running=False, seeded_running=False, seeded_waiting=False) == (False, False)


def test_request_is_seeded() -> None:
    assert request_is_seeded(None) is False
    assert request_is_seeded(_Params(None)) is False
    assert request_is_seeded(_Params(42)) is True
    # An explicit seed of 0 is still an explicit seed.
    assert request_is_seeded(_Params(0)) is True
    assert request_is_seeded(object()) is False
