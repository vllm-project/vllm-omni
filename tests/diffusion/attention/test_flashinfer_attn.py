# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flashinfer_attn import (
    FlashInferAttentionBackend,
    FlashInferAttentionImpl,
)
from vllm_omni.diffusion.data import AttentionSpec, AttnQuantSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _impl(*, causal: bool = False, backend_explicit: bool = False):
    # Avoid CUDA/wrapper init with an initialized-state double.
    obj = FlashInferAttentionImpl.__new__(FlashInferAttentionImpl)
    obj.causal = causal
    obj.softmax_scale = 0.5
    obj.flashinfer_backend = "fa2"
    obj.dtype_qk = None
    obj.dtype_vo = None
    obj.backend_explicit = backend_explicit
    obj._sdpa_fallback = None
    # __new__ bypasses __init__; the mask reject/fallback decisions run inside
    # the boundary op now, so those paths need the op token __init__ would
    # have built.
    obj._op_handle = flashinfer_attn._register_impl_for_boundary_op(obj)
    obj._op_token = torch.tensor(obj._op_handle, dtype=torch.int64)
    return obj


@pytest.mark.cpu
def test_flashinfer_rejects_float_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.zeros(2, 2))

    with pytest.raises(ValueError, match="boolean-only"):
        _impl(backend_explicit=True).forward_cuda(query, query, query, metadata)


@pytest.mark.cpu
def test_flashinfer_rejects_causal_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="causal=True"):
        _impl(causal=True, backend_explicit=True).forward_cuda(query, query, query, metadata)


@pytest.mark.cpu
def test_explicit_cute_dsl_rejects_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=True)
    impl.flashinfer_backend = "cute-dsl"
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="cute-dsl"):
        impl.forward_cuda(query, query, query, metadata)


@pytest.mark.cpu
def test_auto_cute_dsl_falls_back_to_sdpa_for_custom_mask(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=False)
    impl.flashinfer_backend = "cute-dsl"
    impl._run_batch_prefill = lambda *args, **kwargs: pytest.fail("unexpected cute-dsl prefill")
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape


@pytest.mark.cpu
def test_explicit_cute_dsl_is_not_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask() is True
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


@pytest.mark.cpu
def test_explicit_fa2_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


@pytest.mark.cpu
def test_explicit_fa2_backend_kwargs_select_fa2_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert kwargs == {"quant": {"flashinfer_backend": "fa2"}}
    assert FlashInferAttentionImpl._select_backend(requested) == "fa2"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


@pytest.mark.cpu
def test_auto_flashinfer_backend_kwargs_select_cute_dsl_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert "quant" not in kwargs
    assert FlashInferAttentionImpl._select_backend(requested) == "cute-dsl"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 0), "fa2"), ((9, 0), "fa3"), ((10, 0), "cute-dsl"), ((12, 0), "fa2"), ((12, 1), "fa2")],
)
def test_auto_flashinfer_backend_by_capability(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: capability)

    assert FlashInferAttentionImpl._select_backend("auto") == expected
    spec = AttentionSpec(backend="FLASHINFER_ATTN")
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is (expected != "cute-dsl")


@pytest.mark.cpu
def test_hopper_explicit_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flashinfer_real_init_selects_fa2_on_sm80():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real FlashInfer initialization")
    if not flashinfer_attn.HAS_FLASHINFER:
        pytest.skip("FlashInfer is not installed")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("This regression test targets A100/SM80")

    impl = FlashInferAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        backend_kwargs={"quant": {"flashinfer_backend": "auto"}},
    )

    assert impl.flashinfer_backend == "fa2"


@pytest.mark.cpu
def test_valid_kv_length_slices_key_value_prefix_instead_of_masking(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    seen = {}

    def fake_op(query, key, value, mask, token):
        seen.update(key=key.shape, value=value.shape, mask=mask)
        return torch.empty_like(query)

    monkeypatch.setattr(flashinfer_attn, "_flashinfer_attention_op", fake_op)
    query = torch.randn(1, 8, 2, 16)
    metadata = AttentionMetadata(extra={"valid_kv_length": 5})

    _impl(backend_explicit=True).forward_cuda(query, query, query, metadata)

    assert FlashInferAttentionBackend.supports_prefix_kv_slicing
    assert seen == {"key": (1, 5, 2, 16), "value": (1, 5, 2, 16), "mask": None}


@hardware_test(res={"cuda": "B200"}, num_cards=1)
def test_cute_dsl_matches_sdpa_for_strided_value_and_kv_prefix():
    if not torch.cuda.is_available() or not flashinfer_attn.HAS_FLASHINFER:
        pytest.skip("CUDA and FlashInfer are required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("cute-dsl prefill targets SM100")
    torch.manual_seed(0)
    tokens, valid, heads, head_dim = 1031, 1000, 4, 128
    impl = FlashInferAttentionImpl(
        num_heads=heads,
        head_size=head_dim,
        softmax_scale=head_dim**-0.5,
        backend_kwargs={"quant": {"flashinfer_backend": "cute-dsl"}},
    )
    assert impl.flashinfer_backend == "cute-dsl"
    # Q, K and V as strided views of one fused QKV projection, as in the DiT blocks.
    qkv = torch.randn(1, tokens, 3, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    query, key, value = qkv.unbind(2)
    assert not value.is_contiguous()

    output = impl.forward_cuda(query, key, value, AttentionMetadata(extra={"valid_kv_length": valid}))

    expected = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (query, key[:, :valid], value[:, :valid])),
        scale=head_dim**-0.5,
    ).transpose(1, 2)
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("tokens", "heads", "head_dim"), [(2, 3, 64), (257, 56, 128), (64, 7, 96)])
def test_dense_tokens_copies_token_strided_view_exactly(tokens, heads, head_dim):
    qkv = torch.randn(tokens, 3, heads * head_dim, device="cuda", dtype=torch.bfloat16)
    value = qkv[:, 2].view(tokens, heads, head_dim)
    assert not value.is_contiguous()

    dense = flashinfer_attn._dense_tokens(value)

    assert dense.is_contiguous()
    assert torch.equal(dense, value)
