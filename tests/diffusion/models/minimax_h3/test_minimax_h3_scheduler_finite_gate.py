# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_scheduler_default_skips_large_finite_scans_and_preserves_results(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import (
        scheduling_minimax_h3_euler_ancestral as scheduler,
    )

    monkeypatch.delenv("VLLM_OMNI_H3_VALIDATE_FINITE_TENSORS", raising=False)
    original_isfinite = torch.isfinite
    scanned_numels = []

    def record_isfinite(tensor):
        scanned_numels.append(tensor.numel())
        return original_isfinite(tensor)

    monkeypatch.setattr(torch, "isfinite", record_isfinite)
    state = torch.linspace(-1.0, 1.0, 8)
    velocity = torch.linspace(0.25, -0.25, 8)
    timestep = torch.tensor(0.5)
    default_x0 = scheduler.minimax_h3_rf_v_to_x0(state, velocity, timestep)
    default = scheduler.minimax_h3_euler_eta0_step(
        state,
        default_x0,
        sigma_curr=0.5,
        sigma_next=0.25,
    )

    assert scanned_numels == [1]

    checked_x0 = scheduler.minimax_h3_rf_v_to_x0(
        state,
        velocity,
        timestep,
        validate_finite_tensors=True,
    )
    checked = scheduler.minimax_h3_euler_eta0_step(
        state,
        checked_x0,
        sigma_curr=0.5,
        sigma_next=0.25,
        validate_finite_tensors=True,
    )
    torch.testing.assert_close(default_x0, checked_x0, rtol=0, atol=0)
    torch.testing.assert_close(default, checked, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_scheduler_finite_gate_rejects_nonfinite_inputs(monkeypatch, invalid_value):
    from vllm_omni.diffusion.models.minimax_h3 import (
        scheduling_minimax_h3_euler_ancestral as scheduler,
    )

    monkeypatch.setenv("VLLM_OMNI_H3_VALIDATE_FINITE_TENSORS", "1")
    invalid = torch.ones(8)
    invalid[0] = invalid_value

    with pytest.raises(ValueError, match="must be finite"):
        scheduler.minimax_h3_rf_v_to_x0(
            invalid,
            torch.ones_like(invalid),
            torch.tensor(0.5),
        )
    with pytest.raises(ValueError, match="must be finite"):
        scheduler.minimax_h3_euler_eta0_step(
            invalid,
            torch.ones_like(invalid),
            sigma_curr=0.5,
            sigma_next=0.25,
        )
