# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.models.helios.autoencoder_kl_wan import HeliosAutoencoderKLWan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def vae():
    return HeliosAutoencoderKLWan(
        base_dim=4,
        z_dim=2,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
        scale_factor_spatial=2,
        scale_factor_temporal=2,
    ).eval()


@pytest.mark.parametrize("return_dict", [False, True])
@torch.no_grad()
def test_native_decode_reentry(vae, monkeypatch, return_dict):
    monkeypatch.setattr(vae, "is_distributed_enabled", lambda: False)
    first = torch.randn(1, 2, 3, 2, 2)
    for latent in (first, torch.randn(2, 2, 1, 3, 2), first):
        expected = AutoencoderKLWan._decode(vae, latent, return_dict=False)[0]
        result = vae.decode(latent, return_dict=return_dict)
        actual = result.sample if return_dict else result[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(cache is None for cache in vae._feat_map)


@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_all_ranks_receive_video_chunks(vae, monkeypatch, return_dict, rank):
    latent = torch.randn(1, 2, 3, 2, 2)
    video = torch.randn(1, 3, 5, 4, 4)
    local = video if rank == 0 else torch.empty(0)
    monkeypatch.setattr(DistributedAutoencoderKLWan, "tiled_decode", lambda *_a, **_kw: (local,))
    monkeypatch.setattr(vae, "is_distributed_enabled", lambda: True)
    sync = Mock(return_value=video)
    vae.distributed_executor = Mock(_sync_final_result=sync)
    history = []
    vae.clear_cache()
    for _ in range(2):
        vae._feat_map[0] = torch.ones(1)
        result = vae.tiled_decode(latent, return_dict=return_dict)
        assert all(cache is None for cache in vae._feat_map)
        history.append(result.sample if return_dict else result[0])
    sync.assert_called_with(local, 5, latent.device, torch.float32)
    assert sync.call_count == 2
    torch.testing.assert_close(torch.cat(history, dim=2), torch.cat([video, video], dim=2), rtol=0, atol=0)
