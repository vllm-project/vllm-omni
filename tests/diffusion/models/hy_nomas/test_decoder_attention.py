# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage of packed PixelUMM local-attention metadata.

vLLM-Omni's packed varlen contract (``AttentionMetadata.extra`` cu_seqlens) is
emulated with SDPA, so these tests check tile isolation, neighbour order and
batch offsets, not the CUDA kernels.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.hy_nomas import decoder as decoder_module
from vllm_omni.diffusion.models.hy_nomas import decoder_attention
from vllm_omni.diffusion.models.hy_nomas.decoder import PixelUMMDecoder, local_attention_layout

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def attend(query, key, value, scale):
    """[N, heads, D] over [M, kv_heads, D] with grouped-query SDPA."""
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1)[None],
        key.transpose(0, 1)[None],
        value.transpose(0, 1)[None],
        scale=scale,
        enable_gqa=True,
    )
    return output[0].transpose(0, 1)


def packed_varlen(query, key, value, extra):
    """FlashAttention's varlen contract over one packed ``[1, tokens]`` sequence."""
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1
    query = query[0]
    key = key[0]
    value = value[0]
    assert extra["cu_seqlens_q"].dtype == torch.int32
    assert extra["cu_seqlens_k"].dtype == torch.int32
    bounds_q = extra["cu_seqlens_q"].tolist()
    bounds_k = extra["cu_seqlens_k"].tolist()
    assert bounds_q[-1] == query.shape[0]
    assert bounds_k[-1] == key.shape[0]
    scale = query.shape[-1] ** -0.5
    outputs = []
    for segment in range(len(bounds_q) - 1):
        first, last = bounds_q[segment], bounds_q[segment + 1]
        start, end = bounds_k[segment], bounds_k[segment + 1]
        assert last - first <= extra["max_seqlen_q"]
        assert end - start <= extra["max_seqlen_k"]
        outputs.append(attend(query[first:last], key[start:end], value[start:end], scale))
    return torch.cat(outputs).unsqueeze(0)


def eager(neighborhoods, query, key, value, scale):
    """The decoder's per-tile concatenation, one batch element at a time."""
    outputs = []
    for (start, end), neighbors in neighborhoods:
        parts = []
        for index in range(query.shape[0]):
            local_key = torch.cat([key[index, first:last] for first, last in neighbors])
            local_value = torch.cat([value[index, first:last] for first, last in neighbors])
            parts.append(attend(query[index, start:end], local_key, local_value, scale))
        outputs.append(torch.stack(parts))
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize(
    "grid,tile",
    [
        ((4, 16, 20), 4),
        # Partial 2-token edge tiles.
        ((1, 10, 14), 4),
        # PixelUMM's 704x1280 geometry with a short window.
        ((2, 44, 80), 8),
    ],
)
@pytest.mark.parametrize("batch", [1, 2])
def test_packed_plan_matches_per_tile_concatenation(grid, tile, batch):
    _, _, neighborhoods = local_attention_layout(grid, tile, torch.device("cpu"))
    tokens = math.prod(grid)
    heads, kv_heads, dim = 4, 2, 8
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(batch, tokens, heads, dim, generator=generator)
    key = torch.randn(batch, tokens, kv_heads, dim, generator=generator)
    value = torch.randn(batch, tokens, kv_heads, dim, generator=generator)
    plan = decoder_attention.plan_local_attention(neighborhoods, batch, tokens, torch.device("cpu"))
    expected = eager(neighborhoods, query, key, value, dim**-0.5)
    actual = plan(lambda q, k, v, metadata: packed_varlen(q, k, v, metadata.extra), query, key, value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _Backend:
    def __init__(self, supported):
        self.supported = supported

    def supports_multi_doc_packed_varlen(self):
        return self.supported


class _Attention:
    def __init__(self, backend):
        self.attn_backend = backend


def test_packed_attention_declines_unsupported_inputs():
    _, _, neighborhoods = local_attention_layout((1, 16, 16), 4, torch.device("cpu"))
    hidden = torch.zeros(1, 256, 16, dtype=torch.bfloat16)
    capable = _Attention(_Backend(True))
    # CPU tensors keep the eager per-tile path even on a capable backend.
    assert decoder_attention.packed_local_attention(capable, neighborhoods, hidden) is None
    # Backends that cannot isolate N packed documents, and attention without a backend, decline.
    meta = hidden.to("meta")
    assert decoder_attention.packed_local_attention(_Attention(_Backend(False)), neighborhoods, meta) is None
    assert decoder_attention.packed_local_attention(_Attention(None), neighborhoods, meta) is None
    assert decoder_attention.packed_local_attention(capable, neighborhoods, meta.float()) is None


class _SDPA(torch.nn.Module):
    """CPU stand-in for vLLM-Omni attention, including its packed varlen metadata path."""

    attn_backend = None

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, query, key, value, attn_metadata=None):
        if attn_metadata is not None:
            return packed_varlen(query, key, value, attn_metadata.extra)
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=query.shape[-1] ** -0.5,
            enable_gqa=True,
        ).transpose(1, 2)


@torch.no_grad()
@pytest.mark.usefixtures("single_rank_parallel")
def test_decoder_routes_local_attention_through_packed_plan(monkeypatch):
    monkeypatch.setattr(decoder_module, "Attention", _SDPA)
    model = PixelUMMDecoder(
        hidden_size=16,
        intermediate_size=24,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        local_attention_tile_pixels=48,
    ).eval()
    # vLLM's parallel linear layers allocate uninitialized weights.
    generator = torch.Generator().manual_seed(42)
    for parameter in model.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator) * parameter.shape[-1] ** -0.5)
    # Partial edge tiles, two frames and two batch elements exercise offsets.
    latent = torch.randn(2, 32, 2, 2, 5)
    pixels = torch.randn(2, 3, 8, 64, 160)
    expected = model.decoder(latent, pixels)
    plans = []

    def packed(attention, neighborhoods, hidden_states):
        assert isinstance(attention, _SDPA)
        batch, tokens = hidden_states.shape[:2]
        plans.append(decoder_attention.plan_local_attention(neighborhoods, batch, tokens, hidden_states.device))
        return plans[-1]

    monkeypatch.setattr(decoder_module, "packed_local_attention", packed)
    actual = model.decoder(latent, pixels)
    assert len(plans) == 1
    assert plans[0].metadata.extra["cu_seqlens_q"][-1] == 2 * 2 * 4 * 10
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
