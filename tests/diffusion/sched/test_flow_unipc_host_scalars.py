# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for FlowUniPC's host-side scalar math and small-system solve.

``FlowUniPCMultistepScheduler`` used to move 0-d sigma tensors to the accelerator
before doing any algebra with them, then rebuild device tensors from the resulting
device scalars via ``torch.tensor([...])`` -- which has to read their values on the
host, forcing a blocking device-to-host synchronize on every scheduler step. The
chain now stays on the host and the finished coefficient vector is transferred once.

Separately, ``_small_solve`` replaces the general LAPACK solve with Cramer's rule for
the 1x1 and 2x2 systems UniPC actually builds at ``solver_order <= 2``, falling
through to :func:`safe_linalg_solve` for anything larger.

Neither part is meant to change results. The trajectory test gets a genuine
before/after inside one process by pointing ``_small_solve`` at the general solve --
which is exactly the pre-change solve behaviour -- and comparing full denoise
trajectories.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.schedulers import scheduling_flow_unipc_multistep as mod
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
    _small_solve,
)
from vllm_omni.diffusion.utils.flow_matching import safe_linalg_solve

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Measured envelope for the trajectory comparison: max abs error 1.67e-06 at 16 steps
# on values spanning +-3.15, i.e. about 4.4 float32 ULP. Deliberately not
# assert_close at its default rtol -- reordered float32 arithmetic does not reproduce
# bitwise, and pretending otherwise would make this test lie.
TRAJECTORY_ATOL = 5e-06
LATENT_SHAPE = (1, 16, 2, 12, 16)  # DreamZero-ish geometry, shrunk for test runtime


# ── _small_solve ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_small_solve_matches_the_general_solve(n: int) -> None:
    """Cramer's rule for n <= 2, delegation above it, same answer either way."""
    torch.manual_seed(n)
    # Diagonally dominant, so well conditioned at every size.
    matrix = torch.randn(n, n, dtype=torch.float64) + n * torch.eye(n, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    got = _small_solve(matrix, rhs)
    expected = safe_linalg_solve(matrix, rhs)

    assert got.shape == expected.shape == (n,)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2])
def test_small_solve_raises_on_a_singular_system(n: int) -> None:
    """Cramer's rule would return inf/nan; the general solve raises, so this must too."""
    matrix = torch.zeros(n, n, dtype=torch.float32)
    if n == 2:
        # Genuinely singular rather than trivially zero: row 1 duplicates row 0,
        # which is the shape UniPC's Vandermonde R takes when two rk values coincide.
        matrix[0] = torch.tensor([1.0, 2.0])
        matrix[1] = torch.tensor([2.0, 4.0])
    rhs = torch.ones(n, dtype=torch.float32)

    with pytest.raises(torch.linalg.LinAlgError):
        _small_solve(matrix, rhs)


def test_small_solve_delegates_above_two() -> None:
    """The n > 2 path must reach safe_linalg_solve, not a hand-rolled expansion.

    safe_linalg_solve carries a numpy fallback for ROCm wheels whose CPU LAPACK probe
    reports support they do not have. It only triggers for CPU matrices, and this
    change makes the coefficient matrices host tensors -- so losing that delegation
    would break exactly the platform it was added for.
    """
    calls: list[tuple[int, ...]] = []
    real = mod.safe_linalg_solve

    def spy(matrix, rhs):
        calls.append(tuple(matrix.shape))
        return real(matrix, rhs)

    matrix = torch.eye(3, dtype=torch.float64) * 2.0
    rhs = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "safe_linalg_solve", spy)
        got = _small_solve(matrix, rhs)
        # And confirm the small path does NOT delegate.
        _small_solve(torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))

    assert calls == [(3, 3)]
    torch.testing.assert_close(got, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


# ── full denoise trajectory ─────────────────────────────────────────────────────


def _fake_model(sample: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Deterministic stand-in for the DiT: smooth, bounded, and t-dependent."""
    return torch.tanh(sample * 1.3 + t / 1000.0) * 0.9 + sample * 0.05


def _run_denoise(num_steps: int, *, solver_order: int = 2) -> torch.Tensor:
    scheduler = FlowUniPCMultistepScheduler(solver_order=solver_order)
    scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")

    generator = torch.Generator().manual_seed(1234)
    sample = torch.randn(*LATENT_SHAPE, dtype=torch.float32, generator=generator)
    for timestep in scheduler.timesteps:
        sample = scheduler.step(_fake_model(sample, timestep), timestep, sample, return_dict=False)[0]
    return sample


@pytest.mark.parametrize("num_steps", [4, 8, 16])
def test_denoise_trajectory_matches_the_general_solve(num_steps: int) -> None:
    """The closed-form solve must not move the trajectory beyond float32 noise.

    ``_small_solve`` is pointed at the general solve for the reference run, which is
    precisely the pre-change behaviour of both call sites, so this is a real A/B of
    the solve change rather than a comparison against stored data.
    """
    fast = _run_denoise(num_steps)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "_small_solve", safe_linalg_solve)
        reference = _run_denoise(num_steps)

    assert fast.shape == reference.shape == LATENT_SHAPE
    assert torch.isfinite(fast).all(), "closed-form solve produced non-finite latents"
    torch.testing.assert_close(fast, reference, rtol=0.0, atol=TRAJECTORY_ATOL)


@pytest.mark.parametrize("num_steps", [4, 16])
def test_denoise_trajectory_is_finite_at_solver_order_three(num_steps: int) -> None:
    """order=3 builds 3x3 systems, so it exercises the delegation path end to end."""
    out = _run_denoise(num_steps, solver_order=3)
    assert out.shape == LATENT_SHAPE
    assert torch.isfinite(out).all()


def test_sigmas_stay_on_the_host() -> None:
    """The premise of the change: the sigma table is CPU-resident and stays that way.

    ``set_timesteps`` puts ``sigmas`` on the host deliberately. The scalar chain now
    derives from it without an early ``.to(device)``, so a future change that moves
    the table would silently reintroduce the per-step synchronize this removed.
    """
    scheduler = FlowUniPCMultistepScheduler(solver_order=2)
    scheduler.set_timesteps(num_inference_steps=8, device="cpu")
    assert scheduler.sigmas.device.type == "cpu"
