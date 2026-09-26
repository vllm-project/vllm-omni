# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flashinfer_attn import (
    FlashInferAttentionBackend,
    FlashInferAttentionImpl,
)
from vllm_omni.diffusion.data import AttentionSpec, AttnQuantSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _impl(*, causal: bool = False, backend_explicit: bool = False):
    # Avoid CUDA/wrapper init; these tests only cover mask validation.
    obj = FlashInferAttentionImpl.__new__(FlashInferAttentionImpl)
    obj.causal = causal
    obj.softmax_scale = 0.5
    obj.flashinfer_backend = "fa2"
    obj.backend_explicit = backend_explicit
    obj._sdpa_fallback = None
    return obj


def test_flashinfer_rejects_float_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.zeros(2, 2))

    with pytest.raises(ValueError, match="boolean-only"):
        _impl(backend_explicit=True).forward_cuda(query, query, query, metadata)


def test_flashinfer_rejects_causal_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="causal=True"):
        _impl(causal=True, backend_explicit=True).forward_cuda(query, query, query, metadata)


def test_explicit_cute_dsl_rejects_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=True)
    impl.flashinfer_backend = "cute-dsl"
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="cute-dsl"):
        impl.forward_cuda(query, query, query, metadata)


def test_auto_cute_dsl_falls_back_to_sdpa_for_custom_mask(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=False)
    impl.flashinfer_backend = "cute-dsl"
    impl._run_batch_prefill = lambda *args, **kwargs: pytest.fail("unexpected cute-dsl prefill")
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape


def test_explicit_cute_dsl_is_not_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask() is True
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


def test_explicit_fa2_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


def _prims_spec(threshold=None):
    return AttentionSpec(
        backend="FLASHINFER_ATTN",
        quant={"flashinfer_backend": "cute-dsl-prims", "dtype_qk": "fp8_e4m3", "dtype_vo": "fp8_e4m3"},
        skip_softmax={"threshold": threshold} if threshold is not None else None,
    )


@pytest.mark.parametrize("threshold", [None, 0.0, 0.001])
def test_prims_config_and_direct_api_forwarding(monkeypatch, threshold):
    calls = []

    def prefill(q, k, v, o, cu_q, cu_k, *, skip_softmax_threshold=None, **kwargs):
        calls.append((q, k, v, cu_q.clone(), cu_k.clone(), skip_softmax_threshold, kwargs))
        o.fill_(3)

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl.sm120_fmha",
        SimpleNamespace(sm120_fmha_fp8_ragged_prefill=prefill),
    )
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    spec = _prims_spec(threshold)
    impl = FlashInferAttentionImpl(4, 64, 0.125, causal=True, num_kv_heads=2, backend_kwargs=spec.backend_kwargs())
    impl.device = torch.device("cpu")
    assert not FlashInferAttentionBackend.supports_attention_mask(spec)
    # Noncontiguous source tensors must be packed for the direct TMA kernel.
    q = torch.randn(2, 4, 3, 64, dtype=torch.bfloat16).transpose(1, 2)
    kv = torch.randn(2, 2, 5, 64, dtype=torch.bfloat16).transpose(1, 2)
    out = impl.forward_cuda(q, kv, kv)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out, torch.full_like(q, 3))
    flat_q, flat_k, flat_v, cu_q, cu_k, forwarded, kwargs = calls[-1]
    assert flat_q.shape == (6, 4, 64) and flat_k.shape == (10, 2, 64)
    assert all(t.dtype == torch.float8_e4m3fn and t.is_contiguous() for t in (flat_q, flat_k, flat_v))
    assert cu_q.tolist() == [0, 3, 6] and cu_k.tolist() == [0, 5, 10]
    assert cu_q.dtype == cu_k.dtype == torch.int32
    assert forwarded == threshold  # No sequence-length multiplication; preserve zero vs None.
    assert kwargs == {"max_seqlen_q": 3, "is_causal": True, "sm_scale": 0.125}
    old_indptr = impl._qo_indptr
    impl.forward_cuda(q, kv, kv)
    assert impl._qo_indptr is old_indptr
    impl.forward_cuda(q[:, :2], kv[:, :4], kv[:, :4])
    assert calls[-1][3].tolist() == [0, 2, 4]
    assert calls[-1][4].tolist() == [0, 4, 8]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"quant": None}, "requires quant.flashinfer_backend"),
        ({"quant": {"flashinfer_backend": "fa2"}}, "requires quant.flashinfer_backend"),
        ({"quant": {"flashinfer_backend": "cute-dsl-prims"}}, "dtype_qk"),
        ({"skip_softmax": {"target_sparsity": 0.5}}, "absolute skip_softmax.threshold"),
        ({"skip_softmax": {"threshold": float("nan")}}, "finite"),
        ({"skip_softmax": {"threshold": -1}}, "finite"),
    ],
)
def test_prims_config_rejects_unsupported_controls(changes, match):
    kwargs = {
        "backend": "FLASHINFER_ATTN",
        "quant": {"flashinfer_backend": "cute-dsl-prims", "dtype_qk": "fp8_e4m3", "dtype_vo": "fp8_e4m3"},
        "skip_softmax": {"threshold": 0.01},
    }
    kwargs.update(changes)
    with pytest.raises(ValueError, match=match):
        AttentionSpec(**kwargs)


@pytest.mark.parametrize(
    "timestep,expected",
    [(1.0, None), (0.940001, None), (0.94, 0.5), (0.5, 0.5), (0.0, 0.5), (None, None), (float("nan"), None)],
)
def test_prims_timestep_gate_and_runtime_override(monkeypatch, timestep, expected):
    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

    spec = AttentionSpec(
        backend="FLASHINFER_ATTN",
        quant={"flashinfer_backend": "cute-dsl-prims", "dtype_qk": "fp8_e4m3", "dtype_vo": "fp8_e4m3"},
        skip_softmax={"threshold": 0.5, "disabled_until_timestep": 0.94},
    )
    assert spec.backend_kwargs()["disabled_until_timestep"] == 0.94
    impl = _impl()
    impl.skip_softmax_threshold = 0.5
    impl.disabled_until_timestep = 0.94
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with override_forward_context(ForwardContext(denoise_timestep=timestep)):
        assert impl._resolve_sm120_threshold(None) == expected
        runtime = torch.tensor([0.3])
        metadata = AttentionMetadata(extra={"skip_softmax_threshold": runtime})
        resolved = impl._resolve_sm120_threshold(metadata)
        assert resolved is (runtime if expected is not None else None)
        assert impl._resolve_sm120_threshold(AttentionMetadata(extra={"skip_softmax_threshold": None})) is None
    with override_forward_context(None):
        assert impl._resolve_sm120_threshold(None) is None


def test_prims_host_timestep_gate_rejects_graph_capture(monkeypatch):
    impl = _impl()
    impl.skip_softmax_threshold = 0.5
    impl.disabled_until_timestep = 0.94
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="enforce_eager"):
        impl._resolve_sm120_threshold(None)
    impl.disabled_until_timestep = 0.0
    assert impl._resolve_sm120_threshold(None) == 0.5


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (12, 1)])
def test_prims_rejects_unsupported_device(monkeypatch, capability):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    with pytest.raises(ValueError, match="SM120"):
        FlashInferAttentionImpl(4, 64, 0.125, backend_kwargs=_prims_spec().backend_kwargs())


def test_prims_rejects_old_flashinfer_api(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl.sm120_fmha",
        SimpleNamespace(sm120_fmha_fp8_ragged_prefill=lambda q, k, v: None),
    )
    with pytest.raises(ImportError, match="PR #4859"):
        FlashInferAttentionImpl(4, 64, 0.125, backend_kwargs=_prims_spec().backend_kwargs())


def test_prims_rejects_mask_and_ring(monkeypatch):
    from vllm_omni.diffusion.attention.layer import Attention

    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl()
    impl.flashinfer_backend = "cute-dsl-prims"
    q = torch.zeros(1, 2, 2, 64)
    with pytest.raises(ValueError, match="does not support custom masks"):
        impl.forward_cuda(q, q, q, AttentionMetadata(attn_mask=torch.ones(2, 2, dtype=torch.bool)))
    fake = SimpleNamespace(attention=impl, ring_runner=None)
    with pytest.raises(NotImplementedError, match="ring sequence parallelism"):
        Attention._run_ring_attention(fake, q, q, q, None)


def test_prims_capabilities_are_variant_specific(monkeypatch):
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.diffusion.data import AttentionConfig

    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *args: FlashInferAttentionBackend)
    config = AttentionConfig(per_role={"self": _prims_spec(0.001)})
    backend, _ = selector.get_attn_backend_for_role("self", 128, config)
    assert backend.supports_packed_mask_free()
    assert backend.supports_multi_doc_packed_varlen()
    assert not backend.supports_attention_mask()
    assert selector.get_attn_backend_for_capability("self", config) is backend
    ordinary, _ = selector.get_attn_backend_for_role("cross", 128, config)
    assert ordinary is FlashInferAttentionBackend
    assert not ordinary.supports_packed_mask_free()
    assert not ordinary.supports_multi_doc_packed_varlen()


def test_explicit_fa2_backend_kwargs_select_fa2_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert kwargs == {"quant": {"flashinfer_backend": "fa2"}}
    assert FlashInferAttentionImpl._select_backend(requested) == "fa2"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


def test_auto_flashinfer_backend_kwargs_select_cute_dsl_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert "quant" not in kwargs
    assert FlashInferAttentionImpl._select_backend(requested) == "cute-dsl"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


def test_hopper_explicit_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True
