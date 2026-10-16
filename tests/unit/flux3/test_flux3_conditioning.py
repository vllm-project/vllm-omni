# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Flux3 image conditioning helpers."""

import pytest
import torch

from vllm_omni.diffusion.models.flux3.flux3_conditioning import (
    pack_i2i_image_condition,
    pack_image_condition,
    sample_image_latents,
)
from vllm_omni.diffusion.models.flux3.flux3_positional import batched_prc_img, scatter_ids


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scatter_ids_restores_image_latents(dtype):
    images = torch.arange(2 * 4 * 3 * 5).reshape(2, 4, 3, 5).to(dtype)
    tokens, ids = batched_prc_img(images)
    order = torch.randperm(tokens.shape[1], generator=torch.Generator().manual_seed(42))

    restored = scatter_ids(tokens[:, order], ids[:, order])

    assert len(restored) == len(images)
    for actual, expected in zip(restored, images, strict=True):
        assert actual.shape == (1, 4, 3, 5)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected.unsqueeze(0), rtol=0, atol=0)


def test_sample_image_latents_shapes_and_scaled_ids():
    seed = 123
    height, width = 64, 80
    x, x_ids, orig_ids = sample_image_latents(height=height, width=width, seed=seed, device=torch.device("cpu"))

    assert x.shape == (1, 20, 128)
    assert x.dtype == torch.bfloat16
    assert x_ids.shape == (1, 20, 4)
    assert x_ids.dtype == torch.float32
    assert orig_ids.shape == (1, 20, 4)
    assert orig_ids.dtype == torch.long
    torch.testing.assert_close(x_ids[..., 1:3], orig_ids[..., 1:3].float() * 0.5)

    expected = torch.randn((20, 128), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).unsqueeze(0)
    torch.testing.assert_close(x, expected, rtol=0, atol=0)

    rows, cols = torch.meshgrid(torch.arange(height // 16), torch.arange(width // 16), indexing="ij")
    zeros = torch.zeros_like(rows)
    expected_ids = torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(1, 20, 4)
    assert torch.equal(orig_ids, expected_ids)


def test_pack_image_condition_shapes():
    latent = torch.zeros(1, 128, 2, 3)

    x_cond, x_cond_ids, x_cond_timesteps = pack_image_condition(latent, device=torch.device("cpu"))

    assert x_cond.shape == (1, 6, 128)
    assert x_cond.dtype == torch.bfloat16
    assert x_cond_ids.shape == (1, 6, 4)
    assert x_cond_timesteps.shape == (1, 6)
    assert torch.equal(x_cond_timesteps, torch.zeros_like(x_cond_timesteps))


def test_pack_i2i_image_condition_accepts_multiple_references():
    first = torch.arange(128 * 2 * 2, dtype=torch.float32).reshape(128, 2, 2)
    second = torch.ones(128, 2, 2)
    latents = [first, second]

    x_cond, x_cond_ids, x_cond_timesteps = pack_i2i_image_condition(
        latents,
        device=torch.device("cpu"),
        image_rope_scale=0.25,
    )

    assert x_cond.shape == (1, 8, 128)
    assert x_cond_ids.shape == (1, 8, 4)
    assert x_cond_timesteps.shape == (1, 8)
    assert torch.equal(x_cond_timesteps, torch.zeros_like(x_cond_timesteps))
    assert set(x_cond_ids[0, :, 3].tolist()) == {-1.0, -2.0}
    assert x_cond_ids[..., 1:3].max() == 0.25

    expected_first = first.permute(1, 2, 0).reshape(4, 128).to(torch.bfloat16)
    torch.testing.assert_close(x_cond[0, :4], expected_first, rtol=0, atol=0)
    assert torch.equal(x_cond_ids[0, :4, 3], -torch.ones(4))


def test_conditioning_noise_requires_seed():
    latent = torch.zeros(1, 128, 2, 2)

    with pytest.raises(ValueError, match="seed is required"):
        pack_image_condition(latent, device=torch.device("cpu"), conditioning_noise=0.1)
