# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the packed varlen attention path (CPU).

The varlen kernel needs an accelerator, so these tests pin what surrounds it:
the Flash capability rule for quantized policies, the XPU packed dispatch, and
the ``Attention`` layer's ``forward_packed`` entry with its bucketed dense
fallback. Flash's batch-size-1 packed row and the CUDA packed dispatch are
covered on main in ``test_flash_attn.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import vllm_omni.diffusion.attention.backends.utils.fa as fa_module
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
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


def test_flash_capability_rejects_quantized_policies():
    # The FA3 quantized path (forward_fa_quant_cuda) is a dense BSHD kernel:
    # _validate_quant_request_cuda rejects cu_seqlens metadata. The dtype
    # gate runs before any platform or import lookup, so the answer holds on
    # every platform.
    assert FlashAttentionBackend.supports_multi_doc_packed_varlen("fp8") is False


def test_xpu_dispatches_packed_metadata_to_the_packed_path(monkeypatch):
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

    assert impl.forward_xpu(query, key, value, metadata) == "packed"
    assert routed["query"] is query
    assert routed["kwargs"]["cu_seqlens_q"] is metadata.extra["cu_seqlens_q"]


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
    # ``forward_packed`` and its fallbacks call the terminal kernel dispatch;
    # the public ``forward`` must never be re-entered from a packed path.
    object.__setattr__(layer, "_forward_kernel", forward)
    object.__setattr__(layer, "forward", _forbidden_reentry)
    # Quantization-policy attributes consulted by _effective_kv_cache_dtype.
    object.__setattr__(layer, "_kv_cache_dtype", kv_cache_dtype)
    object.__setattr__(layer, "_disable_kv_quant", disable_kv_quant)
    object.__setattr__(layer, "_kv_cache_skip_steps", None)
    object.__setattr__(layer, "_kv_cache_skip_layers", None)
    object.__setattr__(layer, "layer_idx", None)
    return layer


def _forbidden_reentry(*args, **kwargs):
    raise AssertionError("packed attention re-entered Attention.forward")


def test_layer_capability_delegates_to_the_backend_class():
    assert _layer(None, None).supports_multi_doc_packed_varlen() is False
    assert _layer(_DenseBackend, None).supports_multi_doc_packed_varlen() is False
    layer = _layer(_VarlenBackend, None)
    assert layer.supports_multi_doc_packed_varlen() is True
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


# ---------------------------------------------------------------------------
# Attention layer: forward_streams joint entry
# ---------------------------------------------------------------------------


def test_forward_streams_concatenates_in_order_and_splits_back(monkeypatch):
    import vllm_omni.diffusion.attention.layer as layer_module

    calls: list = []
    layer = Attention.__new__(Attention)
    monkeypatch.setattr(Attention, "__call__", lambda self, q, k, v, md=None: calls.append((q, md)) or q * 2)
    monkeypatch.setattr(layer_module, "_armed_mixfusion_plan", lambda: None)
    gen = torch.Generator().manual_seed(0)
    text = tuple(torch.randn(2, 3, HEADS, HEAD_DIM, generator=gen) for _ in range(3))
    image = tuple(torch.randn(2, 5, HEADS, HEAD_DIM, generator=gen) for _ in range(3))
    text_mask = torch.tensor([[True, True, False], [True, True, True]])

    out = layer.forward_streams(
        {"text": text, "image": image}, order=("text", "image"), masks={"text": text_mask, "image": None}
    )

    assert len(calls) == 1
    joint_query, metadata = calls[0]
    assert torch.equal(joint_query, torch.cat([text[0], image[0]], dim=1))
    # A missing stream mask counts as all-valid in the joint key mask.
    assert torch.equal(metadata.attn_mask, torch.cat([text_mask, torch.ones(2, 5, dtype=torch.bool)], dim=1))
    assert torch.equal(out["text"], text[0] * 2)
    assert torch.equal(out["image"], image[0] * 2)


def test_forward_streams_without_masks_passes_no_metadata(monkeypatch):
    import vllm_omni.diffusion.attention.layer as layer_module

    calls: list = []
    layer = Attention.__new__(Attention)
    monkeypatch.setattr(Attention, "__call__", lambda self, q, k, v, md=None: calls.append(md) or q)
    monkeypatch.setattr(layer_module, "_armed_mixfusion_plan", lambda: None)
    qkv = tuple(torch.zeros(1, 2, HEADS, HEAD_DIM) for _ in range(3))

    layer.forward_streams({"a": qkv, "b": qkv}, order=("a", "b"), masks={"a": None, "b": None})

    assert calls == [None]


def test_forward_streams_under_an_armed_step_runs_the_packed_path(monkeypatch):
    import vllm_omni.diffusion.attention.layer as layer_module
    import vllm_omni.diffusion.mixfusion.packed_attention as packed_module

    plan = object()
    seen: list = []
    layer = Attention.__new__(Attention)
    monkeypatch.setattr(Attention, "__call__", _forbidden_reentry)
    monkeypatch.setattr(layer_module, "_armed_mixfusion_plan", lambda: plan)
    monkeypatch.setattr(
        packed_module,
        "run_mixfusion_packed_attention",
        lambda attn, streams, active_plan: seen.append((attn, streams, active_plan)) or {"a": "packed"},
    )
    streams = {"a": (torch.zeros(1), torch.zeros(1), torch.zeros(1))}

    assert layer.forward_streams(streams, order=("a",)) == {"a": "packed"}
    assert seen == [(layer, streams, plan)]
