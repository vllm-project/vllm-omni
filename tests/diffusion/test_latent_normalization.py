# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.utils import (
    denormalize_latents,
    normalize_latents,
    vae_latent_mean_std,
)

CONFIG = SimpleNamespace(
    latents_mean=[0.1, -0.2, 0.3, -0.4],
    latents_std=[1.0, 2.0, 0.5, 1.5],
)


def test_vae_latent_mean_std_shape_and_values():
    mean, std = vae_latent_mean_std(CONFIG, device="cpu", dtype=torch.float32)
    assert mean.shape == (1, 4, 1, 1, 1)
    assert torch.allclose(mean.flatten(), torch.tensor(CONFIG.latents_mean))
    assert torch.allclose(std.flatten(), torch.tensor(CONFIG.latents_std))


def test_vae_latent_mean_std_4d():
    mean, std = vae_latent_mean_std(CONFIG, device="cpu", dtype=torch.float32, ndim=4)
    assert mean.shape == (1, 4, 1, 1)


def test_roundtrip_5d():
    original = torch.randn(2, 4, 3, 8, 8)
    recovered = denormalize_latents(normalize_latents(original, CONFIG), CONFIG)
    assert torch.allclose(original, recovered, atol=1e-6)


def test_roundtrip_4d():
    original = torch.randn(2, 4, 8, 8)
    recovered = denormalize_latents(normalize_latents(original, CONFIG, ndim=4), CONFIG, ndim=4)
    assert torch.allclose(original, recovered, atol=1e-6)


def test_normalize_is_subtract_mean_divide_std():
    latents = torch.randn(1, 4, 1, 1, 1)
    mean, std = vae_latent_mean_std(CONFIG, device="cpu", dtype=torch.float32)
    assert torch.allclose(normalize_latents(latents, CONFIG), (latents - mean) / std)


def test_denormalize_is_multiply_std_add_mean():
    latents = torch.randn(1, 4, 1, 1, 1)
    mean, std = vae_latent_mean_std(CONFIG, device="cpu", dtype=torch.float32)
    assert torch.allclose(denormalize_latents(latents, CONFIG), latents * std + mean)
