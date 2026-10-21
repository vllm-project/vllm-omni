# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for sigma-window attention schedules."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.attention.schedule import (
    parse_attention_sigma_schedule,
    resolve_attention_sigma_schedule,
    resolve_batch_attention_sigma_schedule,
    select_attention_profile_by_sigma,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sigma_schedule_inherits_disables_and_replaces():
    default = parse_attention_sigma_schedule([{"low": 0.0, "high": 0.3, "profile": "sparse"}])
    assert resolve_attention_sigma_schedule(None, default, profiles={"sparse"}) == default
    assert resolve_attention_sigma_schedule([], default, profiles={"sparse"}) == ()
    replaced = [{"low": 0.7, "high": 1.0, "profile": "sparse"}]
    assert resolve_attention_sigma_schedule(replaced, default, profiles={"sparse"})[0].low == 0.7


def test_sigma_selection_uses_normalized_noise_and_gaps():
    schedule = parse_attention_sigma_schedule([{"low": 0.0, "high": 0.3, "profile": "sparse"}])
    assert select_attention_profile_by_sigma(schedule, 0.0) == "sparse"
    assert select_attention_profile_by_sigma(schedule, 0.3) is None
    assert select_attention_profile_by_sigma(schedule, 0.31) is None


def test_batch_resolution_inherits_disables_and_rejects_mixed_windows():
    service = SimpleNamespace(profiles={"sparse": object()}, sigma=[{"low": 0.8, "high": 1.0, "profile": "sparse"}])
    inherit = SimpleNamespace(attention_sigma_schedule=None)
    disable = SimpleNamespace(attention_sigma_schedule=[])
    assert resolve_batch_attention_sigma_schedule([inherit], service)[0].profile == "sparse"
    assert resolve_batch_attention_sigma_schedule([disable], service) == ()
    with pytest.raises(ValueError, match="must be identical"):
        resolve_batch_attention_sigma_schedule([inherit, disable], service)
