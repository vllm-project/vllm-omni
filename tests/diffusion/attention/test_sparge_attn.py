# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from dataclasses import replace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sparge_attn import (
    _SM90_SPARSE_ENTRY_POINT,
    SpargeAttentionBackend,
    SpargeAttentionImpl,
)
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    ParallelStrategy,
    SupportStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture
def fake_ark(monkeypatch):
    package = types.ModuleType("auto_round_kernel")
    calls = []

    def kernel(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return query.clone()

    setattr(package, "sparge_sage2_attn_meansim_topk_xpu", kernel)
    monkeypatch.setitem(sys.modules, "auto_round_kernel", package)
    return package, calls


def test_registry_resolves_sparge_backend():
    assert DiffusionAttentionBackendEnum.SPARGE_ATTN.get_class() is SpargeAttentionBackend
    assert SpargeAttentionBackend.get_name() == "SPARGE_ATTN"
    assert SpargeAttentionBackend.supported_platforms == ("xpu", "cuda")
    assert SpargeAttentionBackend.get_supported_head_sizes() == [64, 128]


@pytest.fixture
def xpu_platform(monkeypatch):
    module = sys.modules["vllm_omni.diffusion.attention.backends.sparge_attn"]
    monkeypatch.setattr(module, "current_omni_platform", types.SimpleNamespace(is_cuda=lambda: False))
    return module


def test_validate_available_reports_missing_package(monkeypatch, xpu_platform):
    monkeypatch.setitem(sys.modules, "auto_round_kernel", None)
    with pytest.raises(ImportError, match="pip install auto-round-lib"):
        SpargeAttentionBackend.validate_available()


def test_validate_available_reports_missing_kernel(monkeypatch, xpu_platform):
    package = types.ModuleType("auto_round_kernel")
    monkeypatch.setitem(sys.modules, "auto_round_kernel", package)
    with pytest.raises(ImportError, match="ARK XPU sparse attention kernel"):
        SpargeAttentionBackend.validate_available()


def test_validate_available_accepts_ark_kernel(fake_ark, xpu_platform):
    SpargeAttentionBackend.validate_available()


@pytest.mark.parametrize(
    ("head_size", "kwargs", "expected"),
    [
        (128, {}, (256, 256, 64)),
        (64, {}, (128, 64, 64)),
        (128, {"sparge_q_block_tokens": 64}, (0, 64, 64)),
        (128, {"sparge_q_block_tokens": 64, "sparge_q_tile_override": 64}, (64, 64, 64)),
    ],
)
def test_head_dimension_defaults(head_size, kwargs, expected):
    impl = SpargeAttentionImpl(8, head_size, head_size**-0.5, num_kv_heads=2, backend_kwargs=kwargs)
    assert (impl.q_tile_override, impl.q_block_tokens, impl.k_block_tokens) == expected


def test_forward_uses_nhd_layout_and_configured_tuning(fake_ark):
    _, calls = fake_ark
    impl = SpargeAttentionImpl(
        8,
        128,
        0.125,
        num_kv_heads=2,
        backend_kwargs={
            "sparge_topk": 0.25,
            "sparge_q_block_tokens": 256,
            "sparge_k_block_tokens": 64,
            "sparge_q_tile_override": 256,
        },
    )
    query = torch.randn(1, 16, 8, 128, dtype=torch.float16)
    key = torch.randn(1, 16, 2, 128, dtype=torch.float16)
    value = torch.randn(1, 16, 2, 128, dtype=torch.float16)
    output = impl.forward_xpu(query, key, value)

    assert output.shape == query.shape
    assert len(calls) == 1
    called_query, called_key, called_value, kwargs = calls[0]
    assert called_query.stride() == query.contiguous().stride()
    assert called_key.stride() == key.contiguous().stride()
    assert called_value.stride() == value.contiguous().stride()
    assert kwargs == {
        "is_causal": False,
        "scale": 0.125,
        "smooth_k": True,
        "simthreshd1": -1.0,
        "topk": 0.25,
        "attention_sink": False,
        "tensor_layout": "NHD",
        "q_tile_override": 256,
        "sparse_q_block_tokens": 256,
        "sparse_k_block_tokens": 64,
    }


def test_forward_compiles_as_custom_op(fake_ark):
    impl = SpargeAttentionImpl(4, 64, 0.125)
    query = torch.randn(1, 16, 4, 64, dtype=torch.float16)
    expected = impl.forward_xpu(query, query, query)
    actual = torch.compile(impl.forward_xpu, fullgraph=True)(query, query, query)
    torch.testing.assert_close(actual, expected)


def test_masks_are_rejected(fake_ark):
    impl = SpargeAttentionImpl(4, 64, 0.125)
    query = torch.randn(1, 16, 4, 64, dtype=torch.float16)
    metadata = AttentionMetadata(attn_mask=torch.ones(1, 16, dtype=torch.bool))
    with pytest.raises(ValueError, match="attn_mask is not supported"):
        impl.forward_xpu(query, query, query, metadata)


def test_packed_metadata_is_rejected(fake_ark):
    impl = SpargeAttentionImpl(4, 64, 0.125)
    query = torch.randn(1, 16, 4, 64, dtype=torch.float16)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 16], dtype=torch.int32)})
    with pytest.raises(ValueError, match="packed variable-length attention"):
        impl.forward_xpu(query, query, query, metadata)


@pytest.mark.parametrize("q_heads,kv_heads,expected", [(8, 8, True), (40, 8, True), (8, 40, False), (8, 3, False)])
def test_cross_head_geometry(q_heads, kv_heads, expected):
    assert SpargeAttentionBackend.supports_cross_head_geometry(q_heads, kv_heads) is expected


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sparge_topk": 0.0}, "top-k ratio"),
        ({"sparge_k_block_tokens": 128}, "k_block_tokens=64"),
        ({"sparge_q_block_tokens": 64, "sparge_q_tile_override": 256}, "requires q_tile_override"),
    ],
)
def test_invalid_tuning_is_rejected(kwargs, match, xpu_platform):
    with pytest.raises(ValueError, match=match):
        SpargeAttentionImpl(8, 128, 0.125, backend_kwargs=kwargs)


def test_invalid_head_dimension_is_rejected():
    with pytest.raises(ValueError, match="supports head sizes"):
        SpargeAttentionImpl(8, 96, 0.125)


def test_valid_execution_path_supports_fullgraph():
    impl = SpargeAttentionImpl(8, 64, 0.125, num_kv_heads=2)
    query = torch.empty(1, 16, 8, 64, dtype=torch.bfloat16)
    key = torch.empty(1, 16, 2, 64, dtype=torch.bfloat16)
    context = ExecutionContext(platform="xpu", require_fullgraph=True)
    result = impl.resolve_execution_path(context, query, key, key, None)
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP
    assert result.requested_support(context).status is SupportStatus.SUPPORTED


def test_mask_execution_path_is_unsupported():
    impl = SpargeAttentionImpl(4, 64, 0.125)
    query = torch.empty(1, 16, 4, 64, dtype=torch.float16)
    metadata = AttentionMetadata(attn_mask=torch.ones(1, 16, dtype=torch.bool))
    result = impl.resolve_execution_path(ExecutionContext(platform="xpu"), query, query, query, metadata)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert result.support.reason == "SPARGE_ATTN: attn_mask is not supported"


def test_mismatched_value_dtype_is_unsupported():
    impl = SpargeAttentionImpl(4, 64, 0.125)
    query = torch.empty(1, 16, 4, 64, dtype=torch.float16)
    value = torch.empty(1, 16, 4, 64, dtype=torch.bfloat16)
    result = impl.resolve_execution_path(ExecutionContext(platform="xpu"), query, query, value, None)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert result.support.reason == "SPARGE_ATTN: Q, K, and V dtypes must match."


@pytest.mark.parametrize("parallel_strategy", [ParallelStrategy.RING, ParallelStrategy.HYBRID_ULYSSES_RING])
def test_ring_execution_path_is_unsupported(parallel_strategy):
    impl = SpargeAttentionImpl(4, 64, 0.125)
    query = torch.empty(1, 16, 4, 64, dtype=torch.float16)
    context = ExecutionContext(platform="xpu", parallel_strategy=parallel_strategy)
    result = impl.resolve_execution_path(context, query, query, query, None)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "ring sequence parallelism" in result.support.reason


@pytest.mark.parametrize(
    "change",
    [
        {"parallel_strategy": ParallelStrategy.ULYSSES},
        {"piecewise": True},
        {"paged_kv": True},
    ],
)
def test_unverified_execution_path_is_not_advertised(change):
    impl = SpargeAttentionImpl(4, 64, 0.125)
    query = torch.empty(1, 16, 4, 64, dtype=torch.float16)
    context = replace(ExecutionContext(platform="xpu"), **change)
    result = impl.resolve_execution_path(context, query, query, query, None)
    assert result.support.status is SupportStatus.UNMIGRATED


@pytest.fixture
def fake_spas_cuda(monkeypatch):
    package = types.ModuleType("spas_sage_attn")
    calls = []

    def kernel(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return query.clone()

    setattr(package, "spas_sage2_attn_meansim_topk_cuda", kernel)
    qattn = types.ModuleType("spas_sage_attn._qattn")
    setattr(qattn, _SM90_SPARSE_ENTRY_POINT, object())
    setattr(package, "_qattn", qattn)
    monkeypatch.setitem(sys.modules, "spas_sage_attn", package)
    monkeypatch.setitem(sys.modules, "spas_sage_attn._qattn", qattn)
    return calls


def test_cuda_packed_padding_is_sliced_before_the_kernel(fake_spas_cuda):
    impl = SpargeAttentionImpl(4, 128, 0.125)
    query = torch.randn(1, 300, 4, 128, dtype=torch.bfloat16)
    key = torch.randn(1, 300, 4, 128, dtype=torch.bfloat16)
    value = torch.randn(1, 300, 4, 128, dtype=torch.bfloat16)
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=200,
            kv_length=200,
            cu_seqlens_q=torch.tensor([0, 200], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 200], dtype=torch.int32),
        ),
        extra={"cu_seqlens_q": torch.tensor([0, 200, 300], dtype=torch.int32)},
    )

    output = impl.forward_cuda(query, key, value, metadata)

    assert output.shape == query.shape
    called_query, called_key, called_value, kwargs = fake_spas_cuda[0]
    assert called_query.shape[1] == 200
    assert called_key.shape[1] == 200
    assert called_value.shape[1] == 200
    assert kwargs["tensor_layout"] == "NHD"
    assert kwargs["topk"] == 0.5
    assert torch.count_nonzero(output[:, 200:]) == 0


def test_cuda_short_query_uses_dense_and_skips_the_sparse_kernel(fake_spas_cuda):
    # The kernel needs >=128 query rows, and zero-padding a shorter query would
    # put pad rows into the first meansim tile and change block selection for
    # the real rows. Short calls must not enter the sparse kernel at all.
    impl = SpargeAttentionImpl(4, 128, 0.125)
    query = torch.randn(1, 64, 4, 128, dtype=torch.float16)
    key = torch.randn(1, 160, 4, 128, dtype=torch.float16)
    value = torch.randn(1, 160, 4, 128, dtype=torch.float16)

    output = impl.forward_cuda(query, key, value)

    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=0.125
    ).transpose(1, 2)
    assert fake_spas_cuda == []
    assert output.shape == query.shape
    torch.testing.assert_close(output, expected)


def test_cuda_raw_cu_seqlens_without_packed_padding_is_rejected(fake_spas_cuda):
    # max_seqlen_* is per document, not a prefix of the packed tensor: slicing it
    # as one prefix would drop later documents or keep padding.
    impl = SpargeAttentionImpl(4, 128, 0.125)
    query = torch.randn(1, 300, 4, 128, dtype=torch.float16)
    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": torch.tensor([0, 100, 120, 300], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 100, 120, 300], dtype=torch.int32),
            "max_seqlen_q": 180,
            "max_seqlen_k": 180,
        },
    )
    with pytest.raises(NotImplementedError, match="packed_padding"):
        impl.forward_cuda(query, query, query, metadata)
    assert fake_spas_cuda == []


def test_cuda_raw_cu_seqlens_execution_path_is_unsupported():
    impl = SpargeAttentionImpl(4, 128, 0.125)
    query = torch.empty(1, 300, 4, 128, dtype=torch.float16)
    metadata = AttentionMetadata(
        extra={"cu_seqlens_q": torch.tensor([0, 100, 200], dtype=torch.int32)},
    )
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, query, query, metadata)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "packed_padding" in result.support.reason


def test_cuda_short_kv_matches_the_kernel_contract(fake_spas_cuda):
    # SpargeAttn only asserts on the query length; short text K/V is legal.
    impl = SpargeAttentionImpl(4, 128, 0.125)
    query = torch.randn(1, 256, 4, 128, dtype=torch.float16)
    key = torch.randn(1, 96, 4, 128, dtype=torch.float16)

    output = impl.forward_cuda(query, key, key)

    assert output.shape == query.shape
    assert fake_spas_cuda[0][1].shape[1] == 96
    assert fake_spas_cuda[0][0].shape[1] == 256


def test_cuda_requires_equal_head_counts(fake_spas_cuda):
    impl = SpargeAttentionImpl(8, 128, 0.125, num_kv_heads=2)
    query = torch.randn(1, 256, 8, 128, dtype=torch.float16)
    key = torch.randn(1, 256, 2, 128, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="equal query and KV head counts"):
        impl.forward_cuda(query, key, key)


def test_cuda_execution_path_is_supported_with_packing():
    impl = SpargeAttentionImpl(4, 128, 0.125)
    query = torch.empty(1, 256, 4, 128, dtype=torch.bfloat16)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=200,
            kv_length=200,
            cu_seqlens_q=torch.tensor([0, 200], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 200], dtype=torch.int32),
        ),
    )
    result = impl.resolve_execution_path(context, query, query, query, metadata)
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP


def test_cuda_gqa_execution_path_is_unsupported():
    impl = SpargeAttentionImpl(8, 128, 0.125, num_kv_heads=2)
    query = torch.empty(1, 256, 8, 128, dtype=torch.float16)
    key = torch.empty(1, 256, 2, 128, dtype=torch.float16)
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, key, key, None)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "equal query and KV head counts" in result.support.reason


def test_cuda_validate_available_requires_spas_sage_attn(monkeypatch):
    module = sys.modules["vllm_omni.diffusion.attention.backends.sparge_attn"]
    monkeypatch.setattr(module, "current_omni_platform", types.SimpleNamespace(is_cuda=lambda: True))
    monkeypatch.setitem(sys.modules, "spas_sage_attn", None)
    with pytest.raises(ImportError, match="SpargeAttn CUDA extension"):
        SpargeAttentionBackend.validate_available()


def test_cuda_validate_available_accepts_spas_sage_attn(monkeypatch, fake_spas_cuda):
    module = sys.modules["vllm_omni.diffusion.attention.backends.sparge_attn"]
    monkeypatch.setattr(module, "current_omni_platform", types.SimpleNamespace(is_cuda=lambda: True))
    SpargeAttentionBackend.validate_available()


def test_cuda_causal_execution_path_is_unsupported():
    impl = SpargeAttentionImpl(4, 128, 0.125, causal=True)
    query = torch.empty(1, 256, 4, 128, dtype=torch.float16)
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), query, query, query, None)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "causal" in result.support.reason


def test_cuda_causal_forward_is_rejected(fake_spas_cuda):
    impl = SpargeAttentionImpl(4, 128, 0.125, causal=True)
    query = torch.randn(1, 256, 4, 128, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="causal"):
        impl.forward_cuda(query, query, query)
    assert fake_spas_cuda == []


def test_cuda_validate_available_requires_sm90_entry_point(monkeypatch, fake_spas_cuda):
    module = sys.modules["vllm_omni.diffusion.attention.backends.sparge_attn"]
    monkeypatch.setattr(module, "current_omni_platform", types.SimpleNamespace(is_cuda=lambda: True))
    monkeypatch.setattr(module, "_cuda_device_capability", lambda: (9, 0))
    qattn = sys.modules["spas_sage_attn._qattn"]
    monkeypatch.delattr(qattn, _SM90_SPARSE_ENTRY_POINT)
    with pytest.raises(ImportError, match="SM90"):
        SpargeAttentionBackend.validate_available()


def test_cuda_validate_available_accepts_sm90_entry_point(monkeypatch, fake_spas_cuda):
    module = sys.modules["vllm_omni.diffusion.attention.backends.sparge_attn"]
    monkeypatch.setattr(module, "current_omni_platform", types.SimpleNamespace(is_cuda=lambda: True))
    monkeypatch.setattr(module, "_cuda_device_capability", lambda: (9, 0))
    SpargeAttentionBackend.validate_available()
