# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
