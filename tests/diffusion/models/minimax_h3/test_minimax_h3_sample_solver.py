# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_res_multistep_matches_comfyui_eta0_equation():
    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import (
        minimax_h3_res_multistep_eta0_step,
    )

    state = torch.tensor([1.0, -2.0], dtype=torch.float32)
    denoised = torch.tensor([0.25, 0.75], dtype=torch.float32)
    old_denoised = torch.tensor([-0.5, 1.25], dtype=torch.float32)
    sigma_prev, sigma_curr, sigma_next = 1.0, 0.7, 0.4

    h = math.log(sigma_curr / sigma_next)
    c2 = math.log(sigma_curr / sigma_prev) / h
    phi1 = math.expm1(-h) / -h
    phi2 = (phi1 - 1.0) / -h
    b1 = phi1 - phi2 / c2
    b2 = phi2 / c2
    expected = math.exp(-h) * state + h * (b1 * denoised + b2 * old_denoised)

    actual = minimax_h3_res_multistep_eta0_step(
        state,
        denoised,
        old_denoised,
        sigma_prev=sigma_prev,
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("old_denoised", "sigma_prev", "sigma_next"),
    [(None, None, 0.4), (torch.tensor([3.0]), 1.0, 0.0)],
)
def test_res_multistep_uses_euler_for_first_and_terminal_steps(old_denoised, sigma_prev, sigma_next):
    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import (
        minimax_h3_euler_eta0_step,
        minimax_h3_res_multistep_eta0_step,
    )

    state = torch.tensor([1.0])
    denoised = torch.tensor([2.0])
    expected = minimax_h3_euler_eta0_step(state, denoised, sigma_curr=0.7, sigma_next=sigma_next)
    actual = minimax_h3_res_multistep_eta0_step(
        state,
        denoised,
        old_denoised,
        sigma_prev=sigma_prev,
        sigma_curr=0.7,
        sigma_next=sigma_next,
    )
    torch.testing.assert_close(actual, expected)


def test_res_multistep_preserves_constant_denoised_solution():
    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import (
        minimax_h3_res_multistep_eta0_step,
    )

    state = torch.tensor([1.5, -0.5])
    denoised = torch.tensor([0.25, 0.75])
    actual = minimax_h3_res_multistep_eta0_step(
        state,
        denoised,
        denoised,
        sigma_prev=1.0,
        sigma_curr=0.7,
        sigma_next=0.4,
    )
    expected = (0.4 / 0.7) * state + (1.0 - 0.4 / 0.7) * denoised
    torch.testing.assert_close(actual, expected)


def test_sample_solver_validation():
    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import (
        minimax_h3_normalize_sample_solver,
    )

    assert minimax_h3_normalize_sample_solver(None) == "euler"
    assert minimax_h3_normalize_sample_solver(" RES_MULTISTEP ") == "res_multistep"
    with pytest.raises(ValueError, match="unsupported MiniMax H3 sample_solver"):
        minimax_h3_normalize_sample_solver("heun")
