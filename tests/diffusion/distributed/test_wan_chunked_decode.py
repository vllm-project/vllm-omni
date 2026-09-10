# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    OmniAutoencoderKLWan,
)
from vllm_omni.diffusion.models.interface import supports_chunked_vae_decode

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def vae():
    # Real causal convolutions and caches, with small random weights for CPU CI.
    return OmniAutoencoderKLWan(
        base_dim=4,
        z_dim=2,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
        scale_factor_spatial=2,
        scale_factor_temporal=2,
    ).eval()


@pytest.mark.parametrize("patch_size", [None, 2])
@pytest.mark.parametrize("return_dict", [False, True])
@torch.no_grad()
def test_chunked_decode_matches_diffusers(vae, patch_size, return_dict):
    if patch_size is not None:
        vae = OmniAutoencoderKLWan.from_config(vae.config, patch_size=patch_size, out_channels=12).eval()
    z = torch.randn(1, 2, 3, 2, 2)
    expected = AutoencoderKLWan._decode(vae, z).sample
    actual = vae.decode(z, return_dict=return_dict)
    torch.testing.assert_close(actual.sample if return_dict else actual[0], expected, rtol=0, atol=0)
    chunks: list[torch.Tensor] = []
    assert supports_chunked_vae_decode(vae)
    assert not supports_chunked_vae_decode(object())
    assert vae.decode_with_chunks(z, on_chunk=chunks.append) is None
    assert [chunk.shape[2] for chunk in chunks] == [1, 2, 2]
    torch.testing.assert_close(torch.cat(chunks, dim=2), expected, rtol=0, atol=0)
    assert all(entry is None for entry in vae._feat_map)


@torch.no_grad()
def test_slicing_and_forward_hooks(vae):
    z = torch.randn(2, 2, 2, 2, 2)
    vae.enable_slicing()
    batches = []
    handle = vae.post_quant_conv.register_forward_pre_hook(lambda module, args: batches.append(args[0].shape[0]))
    vae._hf_hook = SimpleNamespace(pre_forward=Mock())
    try:
        assert vae.decode(z).sample.shape[0] == 2
        assert batches == [1, 1]
        vae._hf_hook.pre_forward.assert_called_once_with(vae)
        batches.clear()
        with pytest.raises(ValueError, match="use_slicing"):
            vae.decode_with_chunks(z, on_chunk=Mock())
        assert batches == []
        vae._hf_hook.pre_forward.reset_mock()
        vae.decode_with_chunks(z[:1], on_chunk=Mock())
        assert batches == [1]
        vae._hf_hook.pre_forward.assert_called_once_with(vae)
    finally:
        handle.remove()


@torch.no_grad()
def test_callback_failure_drains_and_clears_cache(vae):
    z = torch.randn(1, 2, 4, 2, 2)
    decoded = Mock(return_value=None)
    handle = vae.decoder.register_forward_hook(lambda *args: decoded())
    error = RuntimeError("consumer failed")
    callback = Mock(side_effect=error)
    try:
        with pytest.raises(RuntimeError, match="consumer failed") as exc:
            vae.decode_with_chunks(z, on_chunk=callback)
        assert exc.value is error
        assert callback.call_count == 1
        assert decoded.call_count == z.shape[2]
        assert all(entry is None for entry in vae._feat_map)
    finally:
        handle.remove()
    # The same instance can decode again after the failure.
    chunks: list[torch.Tensor] = []
    vae.decode_with_chunks(z, on_chunk=chunks.append)
    torch.testing.assert_close(torch.cat(chunks, 2), vae.decode(z).sample, rtol=0, atol=0)


def test_chunked_decode_rejects_spatial_tiling(vae):
    vae.enable_tiling(tile_sample_min_height=2, tile_sample_min_width=2)
    with pytest.raises(ValueError, match="spatial tiling"):
        vae.decode_with_chunks(torch.zeros(1, 2, 2, 2, 2), on_chunk=Mock())


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fail_callback", [False, True])
def test_spatial_callback_ownership_and_bounded_retention(monkeypatch, rank, fail_callback):
    # Isolate the temporal loop from halo kernels; weakrefs detect retention of
    # decoded GPU-sized tensors even after the consumer has failed.
    monkeypatch.setattr(wan_spatial_shard, "install_wan_spatial_shard_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(wan_spatial_shard, "_rank_world", lambda group: (rank, 2))
    references: list[weakref.ReferenceType[torch.Tensor]] = []

    def decode(x, **kwargs):
        assert sum(ref() is not None for ref in references) <= 1
        chunk = x.clone()
        references.append(weakref.ref(chunk))
        return chunk

    vae = SimpleNamespace(
        config=SimpleNamespace(patch_size=None),
        clear_cache=Mock(),
        post_quant_conv=lambda z: z,
        decoder=decode,
        _feat_map=[],
    )
    callback = Mock(side_effect=RuntimeError("consumer failed") if fail_callback else None)
    z = torch.zeros(1, 3, 6, 2, 2)
    if rank == 0 and fail_callback:
        with pytest.raises(RuntimeError, match="consumer failed"):
            wan_spatial_shard.spatial_shard_decode(vae, z, group=None, on_chunk=callback)
    else:
        assert wan_spatial_shard.spatial_shard_decode(vae, z, group=None, on_chunk=callback) is None
    assert len(references) == z.shape[2]
    assert callback.call_count == (0 if rank else 1 if fail_callback else z.shape[2])
    assert vae.clear_cache.call_count == 2


@pytest.mark.parametrize("rank", [0, 1])
@torch.no_grad()
def test_distributed_small_latent_callback_ownership(vae, monkeypatch, rank):
    vae.is_distributed_enabled = lambda: True
    vae.distributed_executor = SimpleNamespace(group=None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda **kwargs: rank)
    callback = Mock()
    vae.decode_with_chunks(torch.zeros(1, 2, 2, 2, 2), on_chunk=callback)
    assert callback.call_count == (2 if rank == 0 else 0)


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width", "tile"])
def test_distributed_chunked_routing(monkeypatch, mode):
    vae = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    torch.nn.Module.__init__(vae)
    vae.use_tiling = True
    vae.use_slicing = False
    vae.tile_sample_min_height = vae.tile_sample_min_width = 2
    vae.spatial_compression_ratio = 2
    vae.distributed_executor = SimpleNamespace(group=None, parallel_mode=mode, parallel_size=2)
    monkeypatch.setattr(vae, "is_distributed_enabled", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda **kwargs: 2)
    decode = Mock()
    monkeypatch.setattr(wan_spatial_shard, "spatial_shard_decode", decode)
    callback = Mock()
    z = torch.zeros(1, 2, 2, 2, 2)
    if mode == "tile":
        with pytest.raises(ValueError, match="require spatial-shard"):
            vae.decode_with_chunks(z, on_chunk=callback)
        decode.assert_not_called()
    else:
        assert vae.decode_with_chunks(z, on_chunk=callback) is None
        decode.assert_called_once_with(
            vae, z, group=None, return_dict=True, split_dim=mode.removeprefix("spatial_shard_"), on_chunk=callback
        )
