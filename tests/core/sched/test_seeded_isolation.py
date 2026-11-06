# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the seeded-request admission policy.

Seeded TTS requests are only bit-reproducible when they never share an
autoregressive batch (see ``vllm_omni/core/sched/seeded_isolation.py`` for the
measurements). These tests pin the decision table only; the scheduler integration
is exercised by the GPU tests.
"""

import pytest

from vllm_omni.core.sched import seeded_isolation
from vllm_omni.core.sched.seeded_isolation import (
    ENV_VAR,
    isolation_plan,
    request_is_seeded,
    seeded_isolation_enabled,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Params:
    def __init__(self, seed):
        self.seed = seed


@pytest.mark.parametrize(
    "plan",
    [
        # (has_running, seeded_running, seeded_waiting) -> (defer, promote)
        (False, False, False),  # idle, nothing seeded: normal admission
        (False, False, True),  # idle, seeded waiting: admit it alone
        (True, False, False),  # busy, nothing seeded: normal admission
        (True, False, True),  # busy, seeded waiting: wait for the engine to drain
        (True, True, False),  # seeded running: admit nobody else
        (True, True, True),  # seeded running with another seeded waiting
        (False, True, False),  # seeded running alone, nothing waiting
    ],
)
def test_plan_is_off_when_disabled(plan) -> None:
    has_running, seeded_running, seeded_waiting = plan
    assert isolation_plan(
        has_running=has_running,
        seeded_running=seeded_running,
        seeded_waiting=seeded_waiting,
        enabled=False,
    ) == (False, False)


def test_idle_with_seeded_waiting_is_promoted() -> None:
    assert isolation_plan(has_running=False, seeded_running=False, seeded_waiting=True, enabled=True) == (False, True)


def test_seeded_running_defers_everyone() -> None:
    assert isolation_plan(has_running=True, seeded_running=True, seeded_waiting=False, enabled=True) == (True, False)


def test_seeded_waiting_waits_for_running_requests() -> None:
    assert isolation_plan(has_running=True, seeded_running=False, seeded_waiting=True, enabled=True) == (True, False)


def test_unseeded_traffic_is_untouched() -> None:
    assert isolation_plan(has_running=True, seeded_running=False, seeded_waiting=False, enabled=True) == (False, False)
    assert isolation_plan(has_running=False, seeded_running=False, seeded_waiting=False, enabled=True) == (False, False)


def test_request_is_seeded() -> None:
    assert request_is_seeded(None) is False
    assert request_is_seeded(_Params(None)) is False
    assert request_is_seeded(_Params(42)) is True
    # An explicit seed of 0 is still an explicit seed.
    assert request_is_seeded(_Params(0)) is True
    assert request_is_seeded(object()) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "yes", " 1 "])
def test_env_enables(monkeypatch, value) -> None:
    monkeypatch.setenv(ENV_VAR, value)
    assert seeded_isolation_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", ""])
def test_env_disables(monkeypatch, value) -> None:
    monkeypatch.setenv(ENV_VAR, value)
    assert seeded_isolation_enabled() is False


def test_env_default_is_off(monkeypatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert seeded_isolation_enabled() is False


def test_plan_reads_env_when_enabled_is_none(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "1")
    assert seeded_isolation.isolation_plan(has_running=False, seeded_running=False, seeded_waiting=True) == (
        False,
        True,
    )
    monkeypatch.setenv(ENV_VAR, "0")
    assert seeded_isolation.isolation_plan(has_running=False, seeded_running=False, seeded_waiting=True) == (
        False,
        False,
    )
