# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the flat packed varlen FlashAttention path (CPU).

``FlashAttentionImpl._forward_varlen_flat`` runs padding-free FlashAttention over
flat packed ``[total_tokens, num_heads, head_dim]`` Q/K/V. The kernel itself needs
an accelerator, so these tests pin everything around it: the metadata fields the
path consumes, the validation that keeps dense or padded callers off it, and the
dispatch from ``forward_cuda`` / ``forward_xpu``. The kernel call is stubbed so
the packed boundaries it receives can be asserted directly.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import vllm_omni.diffusion.attention.backends.utils.fa as fa_module
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

HEADS, HEAD_DIM = 2, 4
TOTAL_Q, TOTAL_KV = 8, 12


def _impl() -> FlashAttentionImpl:
    return FlashAttentionImpl(num_heads=HEADS, head_size=HEAD_DIM, softmax_scale=HEAD_DIM**-0.5, causal=False)


def _qkv(query_ndim: int = 3):
    query_shape = (TOTAL_Q, HEADS, HEAD_DIM) if query_ndim == 3 else (TOTAL_Q, HEADS, HEAD_DIM, 1)
    return (
        torch.randn(*query_shape),
        torch.randn(TOTAL_KV, HEADS, HEAD_DIM),
        torch.randn(TOTAL_KV, HEADS, HEAD_DIM),
    )


def _varlen_metadata(**overrides: Any) -> AttentionMetadata:
    values: dict[str, Any] = {
        "is_varlen": True,
        "q_cu_seqlens": torch.tensor([0, 3, 8], dtype=torch.int32),
        "kv_cu_seqlens": torch.tensor([0, 5, 12], dtype=torch.int32),
        "max_q_len": 5,
        "max_kv_len": 7,
    }
    values.update(overrides)
    return AttentionMetadata(**values)


def test_varlen_fields_default_to_the_dense_path():
    metadata = AttentionMetadata()

    assert not metadata.is_varlen
    assert metadata.q_cu_seqlens is None
    assert metadata.kv_cu_seqlens is None
    assert metadata.max_q_len is None
    assert metadata.max_kv_len is None
    assert metadata.padded_tokens == 0


@pytest.mark.parametrize("returns_tuple", [False, True])
def test_flat_varlen_passes_packed_boundaries_to_the_kernel(monkeypatch, returns_tuple):
    captured: dict = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        # FA3 returns (out, lse); FA2 returns out.
        return (kwargs["q"], None) if returns_tuple else kwargs["q"]

    monkeypatch.setattr(fa_module, "flash_attn_varlen_func", fake_flash_attn_varlen_func)
    query, key, value = _qkv()
    metadata = _varlen_metadata()

    out = _impl()._forward_varlen_flat(query, key, value, metadata)

    assert captured["q"] is query
    assert captured["k"] is key
    assert captured["v"] is value
    assert captured["cu_seqlens_q"] is metadata.q_cu_seqlens
    assert captured["cu_seqlens_k"] is metadata.kv_cu_seqlens
    assert captured["max_seqlen_q"] == 5
    assert captured["max_seqlen_k"] == 7
    assert captured["causal"] is False
    assert torch.equal(out, query)


@pytest.mark.parametrize(
    ("overrides", "query_ndim", "match"),
    [
        ({}, 4, r"expects \[total_tokens, heads, head_dim\]"),
        ({"attn_mask": torch.zeros(2, 2)}, 3, "does not accept attention masks"),
        ({"joint_attn_mask": torch.zeros(2, 2)}, 3, "does not accept attention masks"),
        ({"padded_tokens": 4}, 3, "requires padded_tokens=0"),
        ({"q_cu_seqlens": None}, 3, "requires q_cu_seqlens and kv_cu_seqlens"),
        ({"kv_cu_seqlens": None}, 3, "requires q_cu_seqlens and kv_cu_seqlens"),
        ({"max_q_len": None}, 3, "requires max_q_len and max_kv_len"),
        ({"max_kv_len": None}, 3, "requires max_q_len and max_kv_len"),
    ],
)
def test_flat_varlen_rejects_anything_but_packed_metadata(overrides, query_ndim, match):
    query, key, value = _qkv(query_ndim)

    with pytest.raises(ValueError, match=match):
        _impl()._forward_varlen_flat(query, key, value, _varlen_metadata(**overrides))


@pytest.mark.parametrize("forward_name", ["forward_cuda", "forward_xpu"])
def test_varlen_metadata_dispatches_to_the_flat_path(monkeypatch, forward_name):
    monkeypatch.setattr(fa_module, "HAS_FLASH_ATTN", True)
    impl = _impl()
    routed: dict = {}

    def fake_forward_varlen_flat(query, key, value, attn_metadata):
        routed["attn_metadata"] = attn_metadata
        return "flat"

    monkeypatch.setattr(impl, "_forward_varlen_flat", fake_forward_varlen_flat)
    query, key, value = _qkv()
    metadata = _varlen_metadata()

    assert getattr(impl, forward_name)(query, key, value, metadata) == "flat"
    assert routed["attn_metadata"] is metadata
