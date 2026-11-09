# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Audex streaming decoder cache (#5964).

The cache used ``torch.cat`` per layer per chunk (quadratic copy traffic) and
RoPE read ``input_pos.max()`` back from the device once per layer. These tests
pin the three fixes: growable per-request buffers, a host-side position, and one
causal mask per chunk instead of one per layer.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.audex.speech_decoder.modeling_audex_causal_speech_decoder import (
    Attention,
    CausalCodecDecoderCache,
    CausalVocosBackbone,
    RotaryPositionalEmbeddings,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _backbone(depth: int = 3) -> CausalVocosBackbone:
    torch.manual_seed(0)
    return CausalVocosBackbone(hidden_dim=32, depth=depth, heads=4, pos_meb_dim=8).eval()


def test_chunked_decode_matches_full_prefill() -> None:
    backbone = _backbone()
    x = torch.randn(1, 7, 32)

    with torch.no_grad():
        full = backbone(x)
        cache = CausalCodecDecoderCache()
        chunked = torch.cat([backbone(x[:, :3], cache=cache), backbone(x[:, 3:], cache=cache)], dim=1)

    assert torch.allclose(full, chunked, atol=1e-5)


def test_token_wise_decode_matches_full_prefill() -> None:
    backbone = _backbone()
    x = torch.randn(1, 6, 32)

    with torch.no_grad():
        full = backbone(x)
        cache = CausalCodecDecoderCache()
        stepwise = torch.cat([backbone(x[:, i : i + 1], cache=cache) for i in range(x.size(1))], dim=1)

    assert torch.allclose(full, stepwise, atol=1e-5)


def test_cache_grows_geometrically_and_keeps_values() -> None:
    cache = CausalCodecDecoderCache()
    keys = [torch.randn(1, 2, 2, 4) for _ in range(8)]

    for key in keys:
        cache.update(0, key, key)

    buffer, _, used = cache.key_values[0]
    assert used == 16
    # Geometric growth: capacity stays under 4x the data, unlike exact-fit growth
    # which would reallocate on every chunk.
    assert buffer.size(2) < 4 * used
    # Only the used prefix is exposed, and it matches the concatenation.
    returned_k, returned_v = cache.update(0, torch.zeros(1, 2, 0, 4), torch.zeros(1, 2, 0, 4))
    assert returned_k.size(2) == used == returned_v.size(2)
    assert torch.allclose(returned_k[:, :, :16], torch.cat(keys, dim=2))


def test_cached_decode_does_not_read_back_from_device(monkeypatch) -> None:
    """``input_pos.max().item()`` was a per-layer device sync in the decode path."""
    backbone = _backbone()
    x = torch.randn(1, 4, 32)

    def _no_readback(_self):
        raise AssertionError("device readback (.item()) on the cached decode path")

    monkeypatch.setattr(torch.Tensor, "item", _no_readback)

    cache = CausalCodecDecoderCache()
    with torch.no_grad():
        backbone(x[:, :2], cache=cache)
        backbone(x[:, 2:], cache=cache)


def test_one_causal_mask_per_chunk_regardless_of_depth(monkeypatch) -> None:
    """The mask is depth-independent, so arange calls must not scale with layers."""
    backbone = _backbone(depth=4)
    x = torch.randn(1, 5, 32)
    cache = CausalCodecDecoderCache()

    calls = {"n": 0}
    original_arange = torch.arange

    def _counting_arange(*args, **kwargs):
        calls["n"] += 1
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", _counting_arange)
    with torch.no_grad():
        backbone(x, cache=cache)
    monkeypatch.setattr(torch, "arange", original_arange)

    # input_positions + the shared key_pos, independent of the 4 layers.
    assert calls["n"] == 2


def test_attention_accepts_an_externally_built_mask() -> None:
    """Passing the shared mask must not change what attention computes."""
    torch.manual_seed(0)
    rotary = RotaryPositionalEmbeddings(dim=8)
    attn = Attention(dim=16, n_heads=2, rotary_embed=rotary, layer_idx=0)
    x = torch.randn(1, 3, 16)
    input_pos = torch.arange(2, 5).unsqueeze(0)

    with torch.no_grad():
        implicit = attn(x, cache=CausalCodecDecoderCache(), input_pos=input_pos)
        key_pos = torch.arange(5).view(1, 1, 1, -1)
        shared = attn(
            x,
            cache=CausalCodecDecoderCache(),
            input_pos=input_pos,
            attn_mask=key_pos <= input_pos.view(1, 1, -1, 1),
            needed_len=5,
        )

    assert torch.allclose(implicit, shared, atol=1e-5)
