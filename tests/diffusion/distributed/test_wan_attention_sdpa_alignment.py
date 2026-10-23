# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks for Wan VAE attention sequence strides.

Cosmos3-Nano decodes with the Wan 2.2 VAE (spatial compression 16, mid-block
channels 1024) and the diffusers tile defaults (256px minimum, 192px stride).
A corner tile whose latent is 1x1 makes ``WanAttentionBlock`` hand SDPA a query
with sequence stride 1. The memory-efficient CUDA kernel rejects that with
``query is not correctly aligned (strideM)``. These tests do not run that
kernel; they check the stride condition and that realignment preserves values.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_sdpa_alignment import (
    align_sdpa_qkv,
    install_wan_attention_sdpa_alignment,
    sdpa_sequence_stride_alignment,
    sequence_stride_is_aligned,
    split_wan_attention_qkv,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# nvidia/Cosmos3-Nano ``vae/config.json`` plus diffusers' Wan tile defaults.
_COSMOS3_SPATIAL_COMPRESSION = 16
_COSMOS3_MID_CHANNELS = 256 * 4
_TILE_LATENT_MIN = 256 // _COSMOS3_SPATIAL_COMPRESSION
_TILE_LATENT_STRIDE = 192 // _COSMOS3_SPATIAL_COMPRESSION
# tests/dfx/stability/tests/test_cosmos3.json samples inclusive integer ranges.
_STABILITY_HEIGHTS = range(720, 1025)
_STABILITY_WIDTHS = range(1024, 1281)


def _tile_lengths(length: int) -> list[int]:
    """Latent extents produced by ``DistributedAutoencoderKLWan.tile_split``."""
    return [min(_TILE_LATENT_MIN, length - start) for start in range(0, length, _TILE_LATENT_STRIDE)]


def _stability_corner_tiles() -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    for height in _STABILITY_HEIGHTS:
        latent_h = height // _COSMOS3_SPATIAL_COMPRESSION
        for width in _STABILITY_WIDTHS:
            latent_w = width // _COSMOS3_SPATIAL_COMPRESSION
            for tile_h in _tile_lengths(latent_h):
                for tile_w in _tile_lengths(latent_w):
                    tiles.add((tile_h, tile_w))
    return tiles


def _qkv_projection(batch_time: int, channels: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch_time, channels * 3, height, width, dtype=dtype, generator=generator)


@pytest.mark.parametrize("channels_last", [False, True])
def test_channels_last_projection_has_the_same_stride_rule(channels_last: bool) -> None:
    projection = _qkv_projection(1, _COSMOS3_MID_CHANNELS, 1, 1, torch.bfloat16)
    if channels_last:
        projection = projection.contiguous(memory_format=torch.channels_last)
    query, key, value = split_wan_attention_qkv(projection)
    assert query.stride(-2) == 1
    aligned_query, aligned_key, aligned_value = align_sdpa_qkv(query, key, value)
    assert sequence_stride_is_aligned(aligned_query)
    assert torch.equal(aligned_query, query)
    assert torch.equal(aligned_key, key)
    assert torch.equal(aligned_value, value)

    wide = _qkv_projection(1, _COSMOS3_MID_CHANNELS, 4, 9, torch.bfloat16)
    if channels_last:
        wide = wide.contiguous(memory_format=torch.channels_last)
    wide_query, _, _ = split_wan_attention_qkv(wide)
    realigned, _, _ = align_sdpa_qkv(wide_query, wide_query, wide_query)
    assert realigned is wide_query


def test_stability_mix_includes_one_by_one_corner_tile() -> None:
    tiles = _stability_corner_tiles()
    assert (1, 1) in tiles
    # Every other edge remainder in this grid keeps a legal sequence stride.
    for tile_h, tile_w in tiles:
        projection = _qkv_projection(1, _COSMOS3_MID_CHANNELS, tile_h, tile_w, torch.float32)
        query, _, _ = split_wan_attention_qkv(projection)
        aligned = sequence_stride_is_aligned(query)
        assert aligned is (tile_h != 1 or tile_w != 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("height", "width"),
    [(1, 1), (1, 16), (16, 1), (16, 16), (9, 4), (4, 8)],
)
def test_align_sdpa_qkv_fixes_only_misaligned_queries(dtype: torch.dtype, height: int, width: int) -> None:
    projection = _qkv_projection(1, _COSMOS3_MID_CHANNELS, height, width, dtype)
    query, key, value = split_wan_attention_qkv(projection)
    aligned_query, aligned_key, aligned_value = align_sdpa_qkv(query, key, value)

    assert sequence_stride_is_aligned(aligned_query)
    assert sequence_stride_is_aligned(aligned_key)
    assert sequence_stride_is_aligned(aligned_value)
    assert torch.equal(aligned_query, query)
    assert torch.equal(aligned_key, key)
    assert torch.equal(aligned_value, value)

    if height == 1 and width == 1:
        alignment = sdpa_sequence_stride_alignment(query)
        assert query.stride(-2) == 1
        assert query.stride(-2) % alignment != 0
        assert aligned_query is not query
        assert aligned_query.stride(-2) % alignment == 0
    else:
        assert aligned_query is query
        assert aligned_key is key
        assert aligned_value is value


class WanAttentionBlock(nn.Module):
    """Stand-in with the diffusers block's attribute names. Not a numeric reference."""

    def __init__(self, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.norm = nn.Identity()
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1).to(dtype=dtype)
        self.proj = nn.Conv2d(dim, dim, 1).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("unpatched Wan attention forward was called")


class _Root(nn.Module):
    def __init__(self, block: WanAttentionBlock) -> None:
        super().__init__()
        self.block = block


def _reference_attention(block: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Diffusers' attention math on the raw, possibly misaligned QKV views."""
    identity = hidden
    batch_size, channels, time, height, width = hidden.size()
    features = hidden.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
    features = block.norm(features)
    query, key, value = split_wan_attention_qkv(block.to_qkv(features))
    features = F.scaled_dot_product_attention(query, key, value)
    features = features.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
    features = block.proj(features)
    features = features.view(batch_size, time, channels, height, width).permute(0, 2, 1, 3, 4)
    return features + identity


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("height", "width"), [(1, 1), (4, 9), (16, 16)])
def test_aligned_forward_matches_unaligned_math(dtype: torch.dtype, height: int, width: int) -> None:
    torch.manual_seed(0)
    block = WanAttentionBlock(32, dtype)
    root = _Root(block)
    install_wan_attention_sdpa_alignment(root)
    hidden = torch.randn(1, 32, 1, height, width, dtype=dtype)

    actual = block(hidden)
    expected = _reference_attention(block, hidden)

    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()
    # A second install must leave the patched forward in place.
    forward = block.forward
    install_wan_attention_sdpa_alignment(root)
    assert block.forward is forward


@pytest.mark.parametrize("alignment_first", [False, True])
def test_alignment_preserves_spatial_shard_gather(monkeypatch, alignment_first: bool) -> None:
    """Both installation orders must retain global attention before resharding."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock as DiffusersWanAttentionBlock

    from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard

    torch.manual_seed(0)
    block = DiffusersWanAttentionBlock(32)
    root = _Root(block)
    local = torch.randn(1, 32, 1, 2, 2)
    remote = torch.randn_like(local)
    full = torch.cat((local, remote), dim=-1)
    expected = block(full)[..., :2]
    calls = []

    def gather(x, *, group, dim):
        calls.append(dim)
        return torch.cat((x, remote), dim=dim)

    monkeypatch.setattr(wan_spatial_shard, "_rank_world", lambda group: (0, 2))
    monkeypatch.setattr(wan_spatial_shard, "all_gather_along_dim", gather)
    if alignment_first:
        install_wan_attention_sdpa_alignment(root)
    wan_spatial_shard._patch_attention_block(block, None, "width")
    sharded_forward = block.forward
    install_wan_attention_sdpa_alignment(root)
    assert block.forward is sharded_forward
    actual = block(local)
    assert calls == [-1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("installed_version", ["0.40.0", "0.41.0", "0.40.0.dev0"])
def test_alignment_warns_on_unverified_diffusers_version(monkeypatch, installed_version: str) -> None:
    """A dependency upgrade must not silently retain the copied forward."""
    from importlib import metadata

    original_version = metadata.version
    monkeypatch.setattr(
        metadata, "version", lambda name: installed_version if name == "diffusers" else original_version(name)
    )
    root = _Root(WanAttentionBlock(32, torch.float32))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        install_wan_attention_sdpa_alignment(root)
        forward = root.block.forward
        install_wan_attention_sdpa_alignment(root)
    assert root.block.forward is forward
    if installed_version == "0.40.0":
        assert not recorded
    else:
        assert len(recorded) == 1
        assert "validated against diffusers 0.40.0" in str(recorded[0].message)
        assert installed_version in str(recorded[0].message)
    hidden = torch.randn(1, 32, 1, 1, 1)
    assert torch.equal(root.block(hidden), _reference_attention(root.block, hidden))
