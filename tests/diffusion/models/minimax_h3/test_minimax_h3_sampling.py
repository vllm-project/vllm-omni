# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numeric regression tests for the reconstructed H3 sampler."""

import json
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver, res_multistep_coeffs
from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import minimax_h3_euler_eta0_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_euler_is_bit_identical(dtype, device):
    # Verify that Euler matches the original update bit-for-bit across devices and dtypes.
    sigmas = [1, 0.9, 0.55, 0.2, 0]
    solver = create_h3_sample_solver("euler", sigmas)
    x = torch.randn(8, 32, dtype=dtype, device=device)
    for i in range(4):
        denoised = torch.randn_like(x)
        expected = minimax_h3_euler_eta0_step(x, denoised, sigma_curr=sigmas[i], sigma_next=sigmas[i + 1])
        x = solver.step(x, denoised, i)
        assert torch.equal(x, expected)


@pytest.mark.parametrize("shift", [1, 3, 12])
@pytest.mark.parametrize("steps", [2, 20, 30, 50])
def test_coefficients_match_frozen_comfyui_formula(shift, steps):
    # Verify that RES coefficients match the frozen ComfyUI reference and sum to the Euler weight.
    reference = json.loads(Path(__file__).with_name("h3_res_multistep_reference.json").read_text())
    case = reference["cases"][f"{shift}_{steps}"]
    sigmas = case["sigmas"]
    coeffs = res_multistep_coeffs(sigmas)
    assert coeffs[0] is None and coeffs[-1] is None
    for i in range(1, steps - 1):
        assert coeffs[i] == pytest.approx(case["coefficients"][i], abs=1e-14)
        assert sum(coeffs[i]) == pytest.approx(1 - sigmas[i + 1] / sigmas[i], abs=1e-14)


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
def test_first_and_last_steps_are_euler(sampler):
    # Verify that the first and final updates use Euler and that history is cleared afterward.
    sigmas = [1, 0.7, 0.25, 0]
    solver = create_h3_sample_solver(sampler, sigmas)
    x = torch.randn(3, 32)
    for i in range(3):
        denoised = torch.randn_like(x)
        expected = minimax_h3_euler_eta0_step(x, denoised, sigma_curr=sigmas[i], sigma_next=sigmas[i + 1])
        x = solver.step(x, denoised, i)
        if i in (0, 2):
            assert torch.equal(x, expected)
    assert solver.old_denoised is None


def test_history_is_private_and_does_not_alias_inputs():
    # Verify that each solver owns independent history that survives mutations of the input tensor.
    sigmas = [1, 0.7, 0.3, 0]
    a = create_h3_sample_solver("res_multistep", sigmas)
    b = create_h3_sample_solver("res_multistep", sigmas)
    x = torch.ones(2, 32)
    d = torch.ones_like(x)
    a.step(x, d, 0)
    d.zero_()
    assert torch.equal(a.old_denoised, x)
    assert b.old_denoised is None
    b.step(x, d, 0)
    assert not torch.equal(a.step(x, d, 1), b.step(x, d, 1))


@pytest.mark.parametrize("sigmas", [[1], [1, float("nan"), 0], [1, 1, 0], [0, 1], [1, -0.1], [1.1, 0]])
def test_invalid_schedule(sigmas):
    # Verify that incomplete, non-finite, non-decreasing, and out-of-range sigma schedules are rejected.
    with pytest.raises(ValueError):
        create_h3_sample_solver("res_multistep", sigmas)


def test_invalid_sampler_and_step_order():
    # Verify that unknown samplers, skipped steps, and repeated steps are rejected.
    with pytest.raises(ValueError):
        create_h3_sample_solver("unknown", [1, 0])
    solver = create_h3_sample_solver("res_multistep", [1, 0.5, 0])
    x = torch.ones(2, 32)
    with pytest.raises(ValueError):
        solver.step(x, x, 1)
    solver.step(x, x, 0)
    with pytest.raises(ValueError):
        solver.step(x, x, 0)
