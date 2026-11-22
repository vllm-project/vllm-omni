# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference seed arithmetic and update rule of the shared fixed-step student sampler."""

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.fixed_step_sde import (
    fixed_step_sample,
    fixed_step_sigmas,
    initial_noise,
    reference_step_seed,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

T_LIST = (1.0, 0.9375, 0.8333333333333334, 0.625)


def test_sigmas_append_terminal_zero_and_validate() -> None:
    assert fixed_step_sigmas(T_LIST) == (*T_LIST, 0.0)
    assert fixed_step_sigmas((*T_LIST, 0.0)) == (*T_LIST, 0.0)
    with pytest.raises(ValueError, match="strictly decreasing"):
        fixed_step_sigmas((1.0, 1.0, 0.5))
    with pytest.raises(ValueError, match="must not be empty"):
        fixed_step_sigmas(())


def test_step_seed_formula_separates_samples_and_steps() -> None:
    assert reference_step_seed(7, 0, 0) == 7 + 9_176
    assert reference_step_seed(7, 3, 2) == 7 + 3 * 1_000_003 + 3 * 9_176
    seeds = {reference_step_seed(0, sample, step) for sample in range(4) for step in range(4)}
    assert len(seeds) == 16
    assert 0 + 0 not in seeds  # initial-noise seed of sample 0 is never reused by a step


def test_sde_matches_inline_reference_update() -> None:
    seed, sample_index = 42, 5
    noise = initial_noise((1, 12), seed=seed, sample_index=sample_index, device="cpu")
    generator = torch.Generator().manual_seed(seed + sample_index)
    assert torch.equal(noise, torch.randn(1, 12, generator=generator))
    weight = torch.randn(12, 12, generator=torch.Generator().manual_seed(1))
    calls: list[float] = []

    def velocity(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        calls.append(float(timestep[0, 0]))
        return x @ weight

    out = fixed_step_sample(
        velocity, noise, t_list=T_LIST, num_train_timesteps=1000, seed=seed, sample_index=sample_index
    )
    sigmas = (*T_LIST, 0.0)
    x = noise.clone()
    for step, (cur, nxt) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        x0 = x - torch.tensor(cur, dtype=torch.float32) * (x @ weight)
        if nxt == 0.0:
            x = x0
            break
        g = torch.Generator().manual_seed(seed + sample_index * 1_000_003 + (step + 1) * 9_176)
        eps = torch.randn(1, 12, generator=g)
        x = (1.0 - torch.tensor(nxt, dtype=torch.float32)) * x0 + torch.tensor(nxt, dtype=torch.float32) * eps
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    # Timesteps are FP32 tensors (``sigma * 1000``), as in the reference.
    assert calls == [float(torch.tensor(sigma * 1000.0, dtype=torch.float32)) for sigma in T_LIST]
    assert out.dtype == torch.float32


def test_ode_uses_velocity_step_without_noise() -> None:
    noise = torch.ones(2, 3)
    out = fixed_step_sample(
        lambda x, t: torch.zeros_like(x),
        noise,
        t_list=(1.0, 0.5),
        num_train_timesteps=1000,
        seed=0,
        sample_index=0,
        sample_type="ode",
    )
    torch.testing.assert_close(out, noise)


def test_rejects_non_flat_state_and_bad_velocity_shape() -> None:
    with pytest.raises(ValueError, match="flat"):
        fixed_step_sample(
            lambda x, t: x, torch.ones(1, 2, 3), t_list=T_LIST, num_train_timesteps=1000, seed=0, sample_index=0
        )
    with pytest.raises(ValueError, match="velocity_fn returned"):
        fixed_step_sample(
            lambda x, t: x[:, :1], torch.ones(1, 4), t_list=T_LIST, num_train_timesteps=1000, seed=0, sample_index=0
        )
