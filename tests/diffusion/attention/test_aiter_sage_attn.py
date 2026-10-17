# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types

import pytest
import torch

from vllm_omni.diffusion.attention.backends import aiter_sage
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    ParallelStrategy,
    SupportStatus,
)
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

B, S, H, D = 1, 64, 4, 128


@pytest.fixture
def kernel(monkeypatch):
    calls = []

    def fake_kernel(query, key, value, softmax_scale):
        assert all(t.is_contiguous() for t in (query, key, value))
        calls.append((query.shape[1], softmax_scale))
        return query + 1

    monkeypatch.setattr(aiter_sage, "_sage_kernel", lambda: fake_kernel)
    return calls


def _impl(**kwargs):
    kwargs.setdefault("causal", False)
    return aiter_sage.AiterSageAttentionImpl(num_heads=H, head_size=D, softmax_scale=0.125, **kwargs)


def _qkv(dtype=torch.bfloat16, head_size=D, seq=S):
    qkv = torch.randn(B, seq, 3, H, head_size, dtype=dtype)
    return qkv.unbind(2)  # strided views, like a fused QKV projection


def _rocm_context(**kwargs):
    return ExecutionContext(platform="rocm", **kwargs)


def test_backend_is_registered():
    backend = DiffusionAttentionBackendEnum.AITER_SAGE.get_class()
    assert backend is aiter_sage.AiterSageAttentionBackend
    assert backend.get_name() == "AITER_SAGE"
    assert backend.get_impl_cls() is aiter_sage.AiterSageAttentionImpl
    assert backend.get_supported_head_sizes() == [D]
    assert backend.supported_platforms == ("rocm",)
    assert backend.supports_prefix_kv_slicing


def test_explicit_selection_rejected_off_rocm():
    class CudaPlatform(OmniPlatform):
        _omni_enum = OmniPlatformEnum.CUDA

    with pytest.raises(ValueError, match="runs on rocm only"):
        CudaPlatform.validate_diffusion_attn_backend("aiter_sage")


@pytest.mark.parametrize("arch", ["gfx942", "gfx1100", "gfx1200"])
def test_validate_available_rejects_other_architectures(monkeypatch, arch):
    monkeypatch.setattr(aiter_sage, "_gcn_arch", lambda: arch)
    with pytest.raises(ValueError, match=f"gfx1201 only, but this GPU is {arch}"):
        aiter_sage.AiterSageAttentionBackend.validate_available()


@pytest.mark.parametrize("has_op", [False, True])
def test_validate_available_requires_aiter_op(monkeypatch, has_op):
    monkeypatch.setattr(aiter_sage, "_gcn_arch", lambda: "gfx1201")
    package = types.ModuleType("aiter")
    if has_op:
        setattr(package, "gfx1201_sage_attention", object())
    monkeypatch.setitem(sys.modules, "aiter", package)
    if has_op:
        aiter_sage.AiterSageAttentionBackend.validate_available()
    else:
        with pytest.raises(ValueError, match="gfx1201_sage_attention"):
            aiter_sage.AiterSageAttentionBackend.validate_available()


def test_dense_forward_makes_inputs_contiguous(kernel):
    q, k, v = _qkv()
    out = _impl().forward_hip(q, k, v)
    assert kernel == [(S, 0.125)]
    torch.testing.assert_close(out, q + 1, rtol=0, atol=0)


@pytest.mark.parametrize("valid", [1, 37, S - 1])
def test_valid_kv_prefix_is_sliced_and_padding_rows_are_zero(kernel, valid):
    q, k, v = _qkv()
    meta = AttentionMetadata(extra={"cu_seqlens_q": None, "valid_kv_length": valid})
    out = _impl().forward_hip(q, k, v, meta)
    assert kernel == [(valid, 0.125)]
    torch.testing.assert_close(out[:, :valid], q[:, :valid] + 1, rtol=0, atol=0)
    assert not out[:, valid:].any()


def test_full_valid_kv_prefix_runs_dense(kernel):
    q, k, v = _qkv()
    _impl().forward_hip(q, k, v, AttentionMetadata(extra={"valid_kv_length": S}))
    assert kernel == [(S, 0.125)]


def _rejections():
    mask = torch.ones(B, S, dtype=torch.bool)
    padding = PackedPaddingMetadata(q_length=S, kv_length=S, cu_seqlens_q=None, cu_seqlens_k=None)
    return [
        pytest.param({"causal": True}, {}, None, "causal", id="causal"),
        pytest.param({"dropout_p": 0.1}, {}, None, "dropout", id="dropout"),
        pytest.param({}, {}, AttentionMetadata(attn_mask=mask), "attn_mask", id="mask"),
        pytest.param({}, {}, AttentionMetadata(extra={"cu_seqlens_q": None}), "packed", id="packed"),
        pytest.param({}, {}, AttentionMetadata(packed_padding=padding), "packed", id="packed_padding"),
        pytest.param({}, {}, AttentionMetadata(extra={"valid_kv_length": S + 1}), "valid_kv_length", id="prefix"),
        pytest.param({}, {"dtype": torch.float16}, None, "BF16", id="fp16"),
        pytest.param({}, {"head_size": 64}, None, "head size", id="head64"),
    ]


@pytest.mark.parametrize(("impl_kwargs", "qkv_kwargs", "meta", "match"), _rejections())
def test_unsupported_inputs_are_rejected(kernel, impl_kwargs, qkv_kwargs, meta, match):
    impl = _impl(**impl_kwargs)
    q, k, v = _qkv(**qkv_kwargs)
    result = impl.resolve_execution_path(_rocm_context(), q, k, v, meta)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert match in result.support.reason
    with pytest.raises(ValueError, match=match):
        impl.forward_hip(q, k, v, meta)
    assert kernel == []


def test_cross_attention_shapes_are_rejected(kernel):
    q, _, _ = _qkv()
    _, k, v = _qkv(seq=S // 2)
    with pytest.raises(ValueError, match="equal shapes"):
        _impl().forward_hip(q, k, v)


def test_dense_rocm_path_is_supported_as_custom_op():
    q, k, v = _qkv()
    meta = AttentionMetadata(extra={"valid_kv_length": S - 3})
    result = _impl().resolve_execution_path(_rocm_context(require_fullgraph=True), q, k, v, meta)
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP
    assert result.requested_support(_rocm_context(require_fullgraph=True)).status is SupportStatus.SUPPORTED


def test_parallel_paths_are_not_claimed():
    q, k, v = _qkv()
    context = _rocm_context(parallel_strategy=ParallelStrategy.ULYSSES)
    result = _impl().resolve_execution_path(context, q, k, v, None)
    assert result.support.status is SupportStatus.UNMIGRATED


@pytest.mark.parametrize("platform", ["cuda", "cpu"])
def test_other_platforms_are_unsupported(platform):
    q, k, v = _qkv()
    result = _impl().resolve_execution_path(ExecutionContext(platform=platform), q, k, v, None)
    assert result.support.status is SupportStatus.UNSUPPORTED
    with pytest.raises(NotImplementedError):
        _impl().forward_cuda(q, k, v)
