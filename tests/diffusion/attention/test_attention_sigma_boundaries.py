# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.attention.schedule import (
    parse_attention_sigma_schedule,
    resolve_request_attention_schedule,
    resolve_request_attention_sigma_schedule,
    select_attention_profile_by_sigma,
    validate_request_attention_schedules,
)
from vllm_omni.diffusion.data import AttentionScheduleConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_adjacent_half_open_windows_and_terminal_one():
    schedule = parse_attention_sigma_schedule(
        [
            {"low": 0.0, "high": 0.3, "profile": "low"},
            {"low": 0.3, "high": 1.0, "profile": "high"},
        ]
    )
    assert [select_attention_profile_by_sigma(schedule, s) for s in [0.0, 0.299, 0.3, 0.99, 1.0]] == [
        "low",
        "low",
        "high",
        "high",
        "high",
    ]


@pytest.mark.parametrize(
    "low,high",
    [
        (-0.01, 0.3),
        (0.1, 1.01),
        (0.3, 0.3),
        (0.4, 0.3),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (False, 0.3),
        # Integers too large for a float must be rejected as client errors, not overflow.
        (0, 10**400),
        (-(10**400), 0.3),
    ],
)
def test_invalid_window_bounds(low, high):
    with pytest.raises((ValueError, TypeError)):
        parse_attention_sigma_schedule([{"low": low, "high": high, "profile": "p"}])


@pytest.mark.parametrize(
    "windows",
    [
        [(0.5, 1.0), (0.0, 0.3)],  # descending
        [(0.0, 0.4), (0.3, 1.0)],  # overlap
    ],
)
def test_unordered_or_overlapping_windows(windows):
    with pytest.raises(ValueError, match="ordered"):
        parse_attention_sigma_schedule([{"low": lo, "high": hi, "profile": "p"} for lo, hi in windows])


def test_service_rejects_combined_defaults():
    with pytest.raises(ValueError, match="cannot combine"):
        AttentionScheduleConfig(
            profiles={"p": {"default": "TORCH_SDPA"}},
            default=[{"start": 0, "end": None, "profile": "p"}],
            sigma=[{"low": 0.0, "high": 1.0, "profile": "p"}],
        )


def test_request_schedule_of_one_kind_replaces_the_server_default_of_the_other():
    profiles = {"p": {"default": "TORCH_SDPA"}}
    step_default = AttentionScheduleConfig(profiles=profiles, default=[{"start": 0, "end": None, "profile": "p"}])
    sigma_default = AttentionScheduleConfig(profiles=profiles, sigma=[{"low": 0.0, "high": 1.0, "profile": "p"}])
    sends_sigma = SimpleNamespace(
        attention_schedule=None, attention_sigma_schedule=[{"low": 0.0, "high": 0.3, "profile": "p"}]
    )
    sends_steps = SimpleNamespace(
        attention_schedule=[{"start": 0, "end": 2, "profile": "p"}], attention_sigma_schedule=None
    )

    validate_request_attention_schedules(sends_sigma, step_default)
    assert resolve_request_attention_schedule(sends_sigma, step_default) == ()
    validate_request_attention_schedules(sends_steps, sigma_default)
    assert resolve_request_attention_sigma_schedule(sends_steps, sigma_default) == ()


def test_request_admission_rejects_both_kinds_and_unknown_profiles():
    service = AttentionScheduleConfig(
        profiles={"p": {"default": "TORCH_SDPA"}},
        default=[{"start": 0, "end": None, "profile": "p"}],
    )
    sampling = SimpleNamespace(
        attention_schedule=[{"start": 0, "end": 2, "profile": "p"}],
        attention_sigma_schedule=[{"low": 0.0, "high": 0.3, "profile": "p"}],
    )
    with pytest.raises(ValueError, match="cannot combine"):
        validate_request_attention_schedules(sampling, service)
    sampling.attention_schedule = []
    validate_request_attention_schedules(sampling, service)
    sampling.attention_sigma_schedule[0]["profile"] = "missing"
    with pytest.raises(ValueError, match="unknown profile"):
        validate_request_attention_schedules(sampling, service)
