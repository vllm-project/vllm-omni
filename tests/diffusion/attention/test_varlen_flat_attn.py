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
from vllm_omni.diffusion.attention.layer import Attention, ScheduleCandidate
from vllm_omni.diffusion.attention.schedule import AttentionScheduleRange
from vllm_omni.diffusion.data import AttentionSpec
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.mixfusion.packed_attention import _backend_supports_attention_mask

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
    def supports_attention_mask(cls, spec=None):
        return False

    @classmethod
    def supports_multi_doc_packed_varlen(cls, kv_cache_dtype: str | None = None) -> bool:
        # Mirrors the Flash rule: the quantized kernel rejects cu_seqlens
        # metadata, so only a dequantized policy keeps the varlen path.
        return kv_cache_dtype in (None, "float", "auto")


class _DenseBackend:
    @classmethod
    def supports_attention_mask(cls, spec=None):
        return spec is not None and spec.backend == "TORCH_SDPA"

    @classmethod
    def supports_multi_doc_packed_varlen(cls, kv_cache_dtype: str | None = None) -> bool:
        return False


def _layer(backend, forward, kv_cache_dtype=None, disable_kv_quant=False):
    layer = Attention.__new__(Attention)
    object.__setattr__(layer, "attn_backend", backend)
    object.__setattr__(layer, "attention", None)
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


def test_packed_routes_follow_the_scheduled_backend_and_restore_baseline():
    """The baseline is varlen; a scheduled dense step must receive isolated buckets."""
    calls = []

    def forward(query, key, value, metadata):
        calls.append((tuple(query.shape), metadata))
        return query

    layer = _layer(_VarlenBackend, forward)
    object.__setattr__(layer, "_schedule_configured", True)
    object.__setattr__(
        layer,
        "_schedule_candidates",
        {
            "dense": ScheduleCandidate(_DenseBackend, AttentionSpec(backend="TORCH_SDPA"), None, True),
        },
    )
    query = torch.randn(8, HEADS, HEAD_DIM)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    for step, masked in [(0, False), (1, True), (2, False)]:
        context = ForwardContext(
            attention_schedule=(AttentionScheduleRange(1, 2, "dense"),),
            denoise_step_idx=step,
            total_denoise_steps=3,
        )
        with override_forward_context(context):
            assert _backend_supports_attention_mask(layer) is masked
            result = layer.forward_packed(query, query, query, cu_seqlens=cu, max_seqlen=5)
        torch.testing.assert_close(result, query)
    assert [shape[:2] for shape, _ in calls] == [(1, 8), (1, 3), (1, 5), (1, 8)]
    assert "cu_seqlens_q" in calls[0][1].extra
    assert calls[1][1].extra == calls[2][1].extra == {}
    assert "cu_seqlens_q" in calls[3][1].extra


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
# Attention layer: MixFusion plan-key dispatch and its re-entry guard
# ---------------------------------------------------------------------------


def _plan_metadata():
    from vllm_omni.diffusion.mixfusion.packed_attention import MIXFUSION_PLAN_KEY, MIXFUSION_STREAMS_KEY

    return AttentionMetadata(extra={MIXFUSION_PLAN_KEY: object(), MIXFUSION_STREAMS_KEY: {}})


def test_plan_key_routes_forward_to_the_packed_path(monkeypatch):
    import vllm_omni.diffusion.attention.layer as layer_module

    seen: list = []
    layer = Attention.__new__(Attention)
    metadata = _plan_metadata()
    monkeypatch.setattr(
        layer_module,
        "run_mixfusion_packed_attention",
        lambda attn, q, k, v, md: seen.append((attn, md)) or {"image": q},
    )
    query = torch.zeros(1, 2, HEADS, HEAD_DIM)

    assert layer.forward(query, query, query, metadata) == {"image": query}
    assert seen == [(layer, metadata)]
    assert layer._mixfusion_packed_active is False


def test_nested_plan_keyed_forward_raises_instead_of_recursing(monkeypatch):
    import vllm_omni.diffusion.attention.layer as layer_module

    layer = Attention.__new__(Attention)
    calls: list = []

    def leaky_packed_path(attn, q, k, v, md):
        # A fallback that wrongly forwards the plan-keyed metadata again.
        calls.append(md)
        return attn.forward(q, k, v, md)

    monkeypatch.setattr(layer_module, "run_mixfusion_packed_attention", leaky_packed_path)
    query = torch.zeros(1, 2, HEADS, HEAD_DIM)

    with pytest.raises(RuntimeError, match="re-entered with MIXFUSION_PLAN_KEY"):
        layer.forward(query, query, query, _plan_metadata())
    assert len(calls) == 1
    # The guard resets on the way out, so the next packed call still works.
    assert layer._mixfusion_packed_active is False


def test_fallback_forward_without_the_plan_key_is_allowed(monkeypatch):
    import vllm_omni.diffusion.attention.layer as layer_module

    layer = Attention.__new__(Attention)
    inner: list = []
    monkeypatch.setattr(Attention, "_forward_impl", lambda self, q, k, v, md: inner.append(md) or q)
    monkeypatch.setattr(Attention, "_uses_hsdp_compile_boundary", lambda self: False)
    monkeypatch.setattr(
        layer_module,
        "run_mixfusion_packed_attention",
        lambda attn, q, k, v, md: {"image": attn.forward(q, k, v, AttentionMetadata())},
    )
    query = torch.ones(1, 2, HEADS, HEAD_DIM)

    out = layer.forward(query, query, query, _plan_metadata())

    assert torch.equal(out["image"], query)
    assert len(inner) == 1 and inner[0].extra == {}
