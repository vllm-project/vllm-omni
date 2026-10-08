# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.attention.schedule import (
    parse_attention_sigma_schedule,
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


def test_request_admission_validates_sigma_and_inherited_step_conflicts():
    service = AttentionScheduleConfig(
        profiles={"p": {"default": "TORCH_SDPA"}},
        default=[{"start": 0, "end": None, "profile": "p"}],
    )
    config = SimpleNamespace(diffusion_attention_schedule=service, cache_backend="none")
    sampling = SimpleNamespace(
        attention_schedule=None, attention_sigma_schedule=[{"low": 0.0, "high": 0.3, "profile": "p"}]
    )
    request = SimpleNamespace(sampling_params=sampling)
    with pytest.raises(ValueError, match="cannot combine"):
        validate_request_attention_schedules(request, config)
    sampling.attention_schedule = []
    validate_request_attention_schedules(request, config)
    sampling.attention_sigma_schedule[0]["profile"] = "missing"
    with pytest.raises(ValueError, match="unknown profile"):
        validate_request_attention_schedules(request, config)
