# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.stable_audio_3.sampling import sample_diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sample(sampler_type: str, seed: int) -> torch.Tensor:
    x = torch.ones(1, 2, 8)
    sigmas = torch.linspace(1.0, 0.0, 5)
    generator = torch.Generator().manual_seed(seed)

    def model(x, t):
        return torch.zeros_like(x)

    return sample_diffusion(model, x, sigmas, sampler_type=sampler_type, generator=generator, disable_tqdm=True)


def test_pingpong_noise_follows_the_request_generator():
    torch.manual_seed(0)
    first = _sample("pingpong", seed=42)
    torch.manual_seed(1)  # the global RNG must not matter
    assert torch.equal(first, _sample("pingpong", seed=42))
    assert not torch.equal(first, _sample("pingpong", seed=43))


def test_deterministic_samplers_do_not_receive_the_generator():
    # The model only accepts (x, t); a leaked generator kwarg would raise.
    _sample("dpmpp-3m-sde", seed=42)
