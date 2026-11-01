# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the packed varlen FlashAttention path (CPU).

``FlashAttentionImpl._forward_varlen_packed`` runs padding-free FlashAttention
over Q/K/V described by the ``AttentionMetadata.extra`` cu_seqlens keys — either
a flat packed ``[total_tokens, num_heads, head_dim]`` layout (dynamic
multi-document batching) or a single ``[1, total_tokens, heads, head_dim]``
packed row. The kernel itself needs an accelerator, so these tests pin
everything around it: the layout handling, the extra-key validation, the
dispatch from ``forward_cuda`` / ``forward_xpu``, and the ``Attention`` layer's
``forward_packed`` entry point with its bucketed dense fallback for backends
without packed varlen support. The kernel call is stubbed so the packed
boundaries it receives can be asserted directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_omni.diffusion.attention.backends.utils.fa as fa_module
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    _normalize_flash_attention_metadata,
)
from vllm_omni.diffusion.attention.layer import Attention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

HEADS, HEAD_DIM = 2, 4
TOTAL_Q, TOTAL_KV = 8, 12


def _impl() -> FlashAttentionImpl:
    return FlashAttentionImpl(num_heads=HEADS, head_size=HEAD_DIM, softmax_scale=HEAD_DIM**-0.5, causal=False)


def _flat_qkv():
    return (
        torch.randn(TOTAL_Q, HEADS, HEAD_DIM),
        torch.randn(TOTAL_KV, HEADS, HEAD_DIM),
        torch.randn(TOTAL_KV, HEADS, HEAD_DIM),
    )


def _packed_metadata(attn_mask: torch.Tensor | None = None) -> AttentionMetadata:
    extra: dict[str, Any] = {
        "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 5, 12], dtype=torch.int32),
        "max_seqlen_q": 5,
        "max_seqlen_k": 7,
    }
    return AttentionMetadata(attn_mask=attn_mask, extra=extra)


@pytest.mark.parametrize("returns_tuple", [False, True])
def test_packed_flat_passes_boundaries_to_the_kernel(monkeypatch, returns_tuple):
    captured: dict = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        # FA3 returns (out, lse); FA2 returns out.
        return (kwargs["q"], None) if returns_tuple else kwargs["q"]

    monkeypatch.setattr(fa_module, "flash_attn_varlen_func", fake_flash_attn_varlen_func)
    query, key, value = _flat_qkv()
    metadata = _packed_metadata()

    out = _impl()._forward_varlen_packed(
        query,
        key,
        value,
        cu_seqlens_q=metadata.extra["cu_seqlens_q"],
        cu_seqlens_k=metadata.extra["cu_seqlens_k"],
        max_seqlen_q=metadata.extra["max_seqlen_q"],
        max_seqlen_k=metadata.extra["max_seqlen_k"],
    )

    assert captured["q"] is query
    assert captured["k"] is key
    assert captured["v"] is value
    assert captured["cu_seqlens_q"] is metadata.extra["cu_seqlens_q"]
    assert captured["cu_seqlens_k"] is metadata.extra["cu_seqlens_k"]
    assert captured["max_seqlen_q"] == 5
    assert captured["max_seqlen_k"] == 7
    assert captured["causal"] is False
    assert torch.equal(out, query)


def test_packed_row_layout_is_flattened_and_restored(monkeypatch):
    captured: dict = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(fa_module, "flash_attn_varlen_func", fake_flash_attn_varlen_func)
    query, key, value = _flat_qkv()
    row_query = query.unsqueeze(0)
    row_key = key.unsqueeze(0)
    row_value = value.unsqueeze(0)
    metadata = _packed_metadata()

    out = _impl()._forward_varlen_packed(
        row_query,
        row_key,
        row_value,
        cu_seqlens_q=metadata.extra["cu_seqlens_q"],
        cu_seqlens_k=metadata.extra["cu_seqlens_k"],
        max_seqlen_q=metadata.extra["max_seqlen_q"],
        max_seqlen_k=metadata.extra["max_seqlen_k"],
    )

    assert captured["q"].ndim == 3
    assert torch.equal(captured["q"], query)
    assert out.shape == row_query.shape
    assert torch.equal(out, row_query)


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        (2, r"expects flat \[total_tokens, heads, head_dim\]"),
        (3, r"expects flat \[total_tokens, heads, head_dim\]"),
    ],
)
def test_packed_rejects_multi_row_layouts(monkeypatch, batch, match):
    monkeypatch.setattr(fa_module, "flash_attn_varlen_func", lambda **kwargs: kwargs["q"])
    query, key, value = _flat_qkv()

    with pytest.raises(ValueError, match=match):
        _impl()._forward_varlen_packed(
            query.unsqueeze(0)
            .expand(batch, -1, -1, -1)
            .reshape(batch * TOTAL_Q, HEADS, HEAD_DIM)
            .view(batch, TOTAL_Q, HEADS, HEAD_DIM),
            key.unsqueeze(0),
            value.unsqueeze(0),
            cu_seqlens_q=torch.tensor([0, TOTAL_Q], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, TOTAL_KV], dtype=torch.int32),
            max_seqlen_q=TOTAL_Q,
            max_seqlen_k=TOTAL_KV,
        )


def test_mixed_ndim_inputs_are_rejected(monkeypatch):
    monkeypatch.setattr(fa_module, "flash_attn_varlen_func", lambda **kwargs: kwargs["q"])
    query, key, value = _flat_qkv()

    with pytest.raises(ValueError, match=r"expects flat \[total_tokens, heads, head_dim\]"):
        _impl()._forward_varlen_packed(
            query.unsqueeze(0),
            key,
            value,
            cu_seqlens_q=torch.tensor([0, TOTAL_Q], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, TOTAL_KV], dtype=torch.int32),
            max_seqlen_q=TOTAL_Q,
            max_seqlen_k=TOTAL_KV,
        )


@pytest.mark.parametrize(
    "drop",
    ["cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"],
)
def test_incomplete_packed_extra_keys_are_rejected(drop):
    extra = {k: v for k, v in _packed_metadata().extra.items() if k != drop}

    with pytest.raises(ValueError, match="Incomplete packed FlashAttention metadata"):
        _normalize_flash_attention_metadata(AttentionMetadata(extra=extra))


def test_flash_capability_rejects_quantized_policies():
    # The FA3 quantized path (forward_fa_quant_cuda) is a dense BSHD kernel:
    # _validate_quant_request_cuda rejects cu_seqlens metadata. The dtype
    # gate runs before any platform or import lookup, so the answer holds on
    # every platform.
    assert FlashAttentionBackend.supports_multi_doc_packed_varlen("fp8") is False


@pytest.mark.parametrize("forward_name", ["forward_cuda", "forward_xpu"])
def test_packed_metadata_dispatches_to_the_packed_path(monkeypatch, forward_name):
    monkeypatch.setattr(fa_module, "HAS_FLASH_ATTN", True)
    impl = _impl()
    routed: dict = {}

    def fake_forward_varlen_packed(query, key, value, **kwargs):
        routed["query"] = query
        routed["kwargs"] = kwargs
        return "packed"

    monkeypatch.setattr(impl, "_forward_varlen_packed", fake_forward_varlen_packed)
    query, key, value = _flat_qkv()
    metadata = _packed_metadata()

    assert getattr(impl, forward_name)(query, key, value, metadata) == "packed"
    assert routed["query"] is query
    assert routed["kwargs"]["cu_seqlens_q"] is metadata.extra["cu_seqlens_q"]


def test_flat_packed_input_with_mask_is_rejected_on_cuda(monkeypatch):
    monkeypatch.setattr(fa_module, "HAS_FLASH_ATTN", True)
    impl = _impl()
    query, key, value = _flat_qkv()
    metadata = _packed_metadata(attn_mask=torch.zeros(1, 1, dtype=torch.bool))

    with pytest.raises(ValueError, match="cannot apply an attention mask"):
        impl.forward_cuda(query, key, value, metadata)


def test_xpu_keeps_the_masked_path_when_a_mask_rides_along(monkeypatch):
    monkeypatch.setattr(fa_module, "HAS_FLASH_ATTN", True)
    impl = _impl()

    def fail_packed(*args, **kwargs):
        raise AssertionError("packed varlen must not run when an attention mask is present")

    monkeypatch.setattr(impl, "_forward_varlen_packed", fail_packed)
    monkeypatch.setattr(
        impl,
        "_forward_varlen_masked",
        lambda query, key, value, mask: "masked",
    )
    query, key, value = _flat_qkv()
    mask = torch.zeros(1, TOTAL_KV, dtype=torch.bool)
    mask[:, 0] = True
    metadata = _packed_metadata(attn_mask=mask)

    assert impl.forward_xpu(query, key, value, metadata) == "masked"


# ---------------------------------------------------------------------------
# Attention layer: forward_packed entry point and bucketed dense fallback
# ---------------------------------------------------------------------------


class _VarlenBackend:
    @classmethod
    def supports_multi_doc_packed_varlen(cls, kv_cache_dtype: str | None = None) -> bool:
        # Mirrors the Flash rule: the quantized kernel rejects cu_seqlens
        # metadata, so only a dequantized policy keeps the varlen path.
        return kv_cache_dtype in (None, "float", "auto")


class _DenseBackend:
    @classmethod
    def supports_multi_doc_packed_varlen(cls, kv_cache_dtype: str | None = None) -> bool:
        return False


def _layer(backend, forward, kv_cache_dtype=None, disable_kv_quant=False):
    layer = Attention.__new__(Attention)
    object.__setattr__(layer, "attn_backend", backend)
    object.__setattr__(layer, "forward", forward)
    # Quantization-policy attributes consulted by _effective_kv_cache_dtype.
    object.__setattr__(layer, "_kv_cache_dtype", kv_cache_dtype)
    object.__setattr__(layer, "_disable_kv_quant", disable_kv_quant)
    object.__setattr__(layer, "_kv_cache_skip_steps", None)
    object.__setattr__(layer, "_kv_cache_skip_layers", None)
    object.__setattr__(layer, "layer_idx", None)
    return layer


def test_layer_capability_delegates_to_the_backend_class():
    assert _layer(None, None).supports_multi_doc_packed_varlen() is False
    assert _layer(_VarlenBackend, None).supports_multi_doc_packed_varlen() is True
    assert _layer(_DenseBackend, None).supports_multi_doc_packed_varlen() is False


def test_layer_capability_passes_the_quantization_policy():
    layer = _layer(_VarlenBackend, None)
    assert layer.supports_multi_doc_packed_varlen("float") is True
    assert layer.supports_multi_doc_packed_varlen("auto") is True
    assert layer.supports_multi_doc_packed_varlen("fp8") is False


def test_forward_packed_routes_capability_to_one_varlen_forward():
    calls: list = []

    def fake_forward(query, key, value, attn_metadata):
        calls.append((query, key, value, attn_metadata))
        return query

    layer = _layer(_VarlenBackend, fake_forward)
    query, key, value = _flat_qkv()
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    out = layer.forward_packed(query, key, value, cu_seqlens=cu_seqlens, max_seqlen=8)

    assert out.shape == query.shape
    assert torch.equal(out, query)
    assert len(calls) == 1
    called_query, called_key, called_value, metadata = calls[0]
    # The native path hands the backend a [1, total_tokens, heads, head_dim]
    # packed row (TRTLLM's forward_cuda unpacks a 4-D query) and restores the
    # flat packing on the way out.
    assert called_query.shape == (1, TOTAL_Q, HEADS, HEAD_DIM)
    assert torch.equal(called_query, query.unsqueeze(0))
    assert torch.equal(called_key, key.unsqueeze(0))
    assert torch.equal(called_value, value.unsqueeze(0))
    assert metadata.attn_mask is None
    assert torch.equal(metadata.extra["cu_seqlens_q"], cu_seqlens)
    assert torch.equal(metadata.extra["cu_seqlens_k"], cu_seqlens)
    assert metadata.extra["max_seqlen_q"] == 8
    assert metadata.extra["max_seqlen_k"] == 8


def _segment_mean_flat(query, key, value, cu_seqlens):
    """Reference semantics: each document adds its own K/V means to its rows."""
    out = torch.empty_like(query)
    lengths = cu_seqlens.diff().tolist()
    offset = 0
    for length in lengths:
        if length:
            out[offset : offset + length] = (
                query[offset : offset + length]
                + key[offset : offset + length].mean(dim=0, keepdim=True)
                + value[offset : offset + length].mean(dim=0, keepdim=True)
            )
        offset += length
    return out


def test_forward_packed_dense_buckets_isolate_documents():
    dense_shapes: list = []

    def dense_forward(query, key, value, attn_metadata):
        assert attn_metadata is not None and attn_metadata.attn_mask is None
        dense_shapes.append(tuple(query.shape))
        return query + key.mean(dim=1, keepdim=True) + value.mean(dim=1, keepdim=True)

    layer = _layer(_DenseBackend, dense_forward)
    gen = torch.Generator().manual_seed(7)
    total = 10
    query = torch.randn(total, HEADS, HEAD_DIM, generator=gen)
    key = torch.randn(total, HEADS, HEAD_DIM, generator=gen)
    value = torch.randn(total, HEADS, HEAD_DIM, generator=gen)
    # Documents of length 4, 2 and 4: the two length-4 documents share one
    # dense bucket, the length-2 document gets its own.
    cu_seqlens = torch.tensor([0, 4, 6, 10], dtype=torch.int32)

    out = layer.forward_packed(query, key, value, cu_seqlens=cu_seqlens, max_seqlen=4)

    assert torch.equal(out, _segment_mean_flat(query, key, value, cu_seqlens))
    # Equal lengths are bucketed together: [2, 4, H, D] and [1, 2, H, D].
    assert sorted(dense_shapes) == [(1, 2, HEADS, HEAD_DIM), (2, 4, HEADS, HEAD_DIM)]


def test_forward_packed_dense_buckets_skip_empty_documents():
    def dense_forward(query, key, value, attn_metadata):
        assert query.shape[0] == 2 and query.shape[1] == 4
        return query

    layer = _layer(_DenseBackend, dense_forward)
    total = 8
    query = torch.arange(total * HEADS * HEAD_DIM, dtype=torch.float32).reshape(total, HEADS, HEAD_DIM)
    # A zero-length document between two length-4 documents.
    cu_seqlens = torch.tensor([0, 4, 4, 8], dtype=torch.int32)

    out = layer.forward_packed(query, query, query, cu_seqlens=cu_seqlens, max_seqlen=4)

    assert torch.equal(out, query)


def test_quantized_policy_falls_back_to_dense_buckets():
    # A non-float effective quantization policy cannot ride the varlen
    # kernel (the FA3 quantized path rejects cu_seqlens metadata), so the
    # dispatcher must serve it through the dense bucket fallback instead of
    # crashing inside the quantized kernel.
    dense_shapes: list = []

    def dense_forward(query, key, value, attn_metadata):
        assert "cu_seqlens_q" not in attn_metadata.extra
        dense_shapes.append(tuple(query.shape))
        return query + key.mean(dim=1, keepdim=True) + value.mean(dim=1, keepdim=True)

    layer = _layer(_VarlenBackend, dense_forward, kv_cache_dtype="fp8")
    query, key, value = _flat_qkv()
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

    out = layer.forward_packed(query, key, value, cu_seqlens=cu_seqlens, max_seqlen=4)

    assert torch.equal(out, _segment_mean_flat(query, key, value, cu_seqlens))
    # The two length-4 documents share one equal-length dense bucket.
    assert dense_shapes == [(2, 4, HEADS, HEAD_DIM)]


def test_disabled_quant_policy_keeps_the_varlen_path():
    # _disable_kv_quant makes the effective policy "float", which the varlen
    # path accepts.
    calls: list = []

    def varlen_forward(query, key, value, attn_metadata):
        calls.append((query.shape, attn_metadata))
        return query

    layer = _layer(_VarlenBackend, varlen_forward, kv_cache_dtype="fp8", disable_kv_quant=True)
    query, key, value = _flat_qkv()
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    out = layer.forward_packed(query, key, value, cu_seqlens=cu_seqlens, max_seqlen=8)

    assert len(calls) == 1
    assert calls[0][0] == (1, TOTAL_Q, HEADS, HEAD_DIM)
    assert torch.equal(calls[0][1].extra["cu_seqlens_q"], cu_seqlens)
    assert out.shape == query.shape
    assert torch.equal(out, query)


def test_forward_packed_accepts_the_single_packed_row_layout():
    def varlen_forward(query, key, value, attn_metadata):
        assert query.ndim == 4 and query.shape[0] == 1
        return query

    layer = _layer(_VarlenBackend, varlen_forward)
    query, key, value = _flat_qkv()

    out = layer.forward_packed(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        cu_seqlens=torch.tensor([0, TOTAL_Q], dtype=torch.int32),
        max_seqlen=TOTAL_Q,
    )

    assert out.shape == (1, TOTAL_Q, HEADS, HEAD_DIM)
    assert torch.equal(out, query.unsqueeze(0))


def test_forward_packed_rejects_multi_row_layouts():
    layer = _layer(_VarlenBackend, None)

    with pytest.raises(ValueError, match=r"expects a flat or \[1, total_tokens"):
        layer.forward_packed(
            torch.randn(2, 4, HEADS, HEAD_DIM),
            torch.randn(2, 4, HEADS, HEAD_DIM),
            torch.randn(2, 4, HEADS, HEAD_DIM),
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            max_seqlen=4,
        )


def test_real_layer_fallback_uses_attention_forward():
    # The production fallback routes every bucket through the real
    # ``Attention.forward`` (validation included); the stub only stands in for
    # the underlying dense backend.
    captured: list = []

    def dense_forward(query, key, value, attn_metadata):
        captured.append((tuple(query.shape), attn_metadata))
        return query

    fake_layer = SimpleNamespace(forward=dense_forward)
    query = torch.randn(6, HEADS, HEAD_DIM)
    cu_seqlens = torch.tensor([0, 2, 4, 6], dtype=torch.int32)

    out = Attention._forward_packed_dense_buckets(fake_layer, query, query, query, cu_seqlens)

    assert torch.equal(out, query)
    assert [shape for shape, _ in captured] == [(3, 2, HEADS, HEAD_DIM)]
