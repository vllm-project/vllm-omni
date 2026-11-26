# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.physis.ltx.sampler import LTXFourStepSampler

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def test_ltx_four_step_shift_five_ladder():
    sampler = LTXFourStepSampler()
    # Pinned WaveServe grid: the terminal 1/1000 point is shifted before linspace.
    terminal = 5.0 * 0.001 / (1.0 + 4.0 * 0.001)
    unshifted = torch.linspace(1.0, terminal, 4, dtype=torch.float64)
    expected = (5.0 * unshifted / (1.0 + 4.0 * unshifted)).float()
    assert sampler.sigmas == (*expected.tolist(), 0.0)
    assert sampler.timesteps == tuple(1000.0 * value for value in expected.tolist())


@pytest.mark.parametrize("chunk", [0, 1, 13])
def test_noise_and_every_transition_match_reference(chunk):
    sampler = LTXFourStepSampler()
    shape = (1, 128, 3 if chunk == 0 else 1, 2, 3)
    seed = 17
    initial_generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + chunk * 4096)
    expected = torch.randn(shape, generator=initial_generator, dtype=torch.float32)
    actual = sampler.initial_noise(shape, device="cpu", seed=seed, chunk=chunk)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for step in range(4):
        prediction = torch.linspace(-1, 1, actual.numel()).reshape(shape).to(torch.bfloat16)
        clean = expected.float() - sampler.sigmas[step] * prediction.float()
        following = sampler.sigmas[step + 1]
        if following:
            generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + chunk * 4096 + step + 1)
            noise = torch.randn(shape, generator=generator, dtype=torch.float32)
            expected = (1.0 - following) * clean + following * noise
        else:
            expected = clean
        actual = sampler.advance(actual, prediction, step=step, seed=seed, chunk=chunk)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
