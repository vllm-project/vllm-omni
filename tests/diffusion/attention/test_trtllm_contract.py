# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends import trtllm_attn as tg
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    OuterBoundary,
    ParallelStrategy,
    SupportStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _impl(**kwargs):
    return tg.TrtllmAttentionImpl(2, 128, 128**-0.5, **kwargs)


def _inputs(q_len=8, kv_len=12, dtype=torch.bfloat16):
    return (
        torch.randn(1, q_len, 2, 128, dtype=dtype),
        torch.randn(1, kv_len, 2, 128, dtype=dtype),
        torch.randn(1, kv_len, 2, 128, dtype=dtype),
    )


@pytest.fixture
def dispatcher(monkeypatch, tmp_path):
    marker = tmp_path / "kernel"
    marker.write_text("loaded")

    @functools.cache
    def load_kernel():
        return marker.read_text()

    def fake_attention(query, key, value, workspace_buffer, batch_size, bmm1_scale, **kwargs):
        assert load_kernel() == "loaded"
        assert kwargs["skip_all_rows_active_check"] is True
        workspace_buffer.add_(1)
        q, k, v = (t.reshape(batch_size, -1, t.shape[-2], t.shape[-1]).transpose(1, 2) for t in (query, key, value))
        return (
            F.scaled_dot_product_attention(q, k, v, scale=bmm1_scale)
            .transpose(1, 2)
            .reshape(query.shape[0], query.shape[1], value.shape[-1])
            .contiguous()
        )

    def quantize(q, k, v, **kwargs):
        scale = torch.ones(1)
        return q, k, v, scale, scale, scale, None

    # CPU tests substitute device selection; the real kernel tests use the GPU.
    monkeypatch.setattr(tg, "_is_trtllm_blackwell_device", lambda query: True)
    monkeypatch.setattr(tg, "HAS_FLASHINFER", True)
    monkeypatch.setattr(tg, "trtllm_ragged_attention_deepseek", fake_attention, raising=False)
    monkeypatch.setattr(
        tg, "flashinfer", SimpleNamespace(__version__="0.6.18", trtllm_sage_attention_quantize=quantize), raising=False
    )
    workspace = torch.zeros(16, dtype=torch.uint8)
    monkeypatch.setattr(tg.TrtllmAttentionImpl, "_get_workspace", classmethod(lambda cls, device: workspace))
    torch.compiler.reset()
    yield workspace
    torch.compiler.reset()


def test_dense_contract_uses_initialized_state(dispatcher):
    context = ExecutionContext(platform="cuda", kernel_variant="caller_sage", causal=True, require_fullgraph=True)
    result = _impl().resolve_execution_path(context, *_inputs(), None)
    assert result.path == "trtllm_dense" and result.kernel_variant == "trtllm"
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP
    assert result.requested_support(context).status is SupportStatus.SUPPORTED
    pre = tg.TrtllmAttentionBackend.resolve_capabilities(context)
    assert pre.kernel_variant is None and pre.support.status is SupportStatus.UNMIGRATED
    assert pre.requested_support(context).status is SupportStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "change",
    [
        {"platform": "rocm"},
        {"paged_kv": True},
        {"parallel_strategy": ParallelStrategy.ULYSSES},
        {"parallel_strategy": ParallelStrategy.RING},
        {"parallel_strategy": ParallelStrategy.HYBRID_ULYSSES_RING},
        {"parallel_strategy": ParallelStrategy.ALLGATHER_KV},
        {"outer_boundaries": frozenset({OuterBoundary.HSDP})},
    ],
)
def test_outer_paths_are_unmigrated(dispatcher, change):
    context = replace(ExecutionContext(platform="cuda", require_fullgraph=True), **change)
    result = _impl().resolve_execution_path(context, *_inputs(), None)
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "kwargs,path",
    [
        ({"causal": True}, "trtllm_dense"),
        ({"backend_kwargs": {"quant": {"dtype_qk": "int8"}}}, "trtllm_sage"),
        ({"backend_kwargs": {"skip_softmax_threshold": 0.01}}, "trtllm_skip_softmax"),
        ({"backend_kwargs": {"target_sparsity": 0.5}}, "trtllm_skip_softmax"),
    ],
)
def test_unverified_backend_settings(dispatcher, kwargs, path):
    result = _impl(**kwargs).resolve_execution_path(ExecutionContext(platform="cuda"), *_inputs(), None)
    assert result.path == path and result.support.status is SupportStatus.UNMIGRATED


@pytest.mark.parametrize("padding", [False, True])
def test_packed_paths_are_not_dense(dispatcher, padding):
    q, k, v = _inputs()
    cuq, cuk = torch.tensor([0, 8], dtype=torch.int32), torch.tensor([0, 12], dtype=torch.int32)
    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": cuq,
            "cu_seqlens_k": cuk,
            "max_seqlen_q": 8,
            "max_seqlen_k": 12,
        },
        packed_padding=PackedPaddingMetadata(8, 12, cuq, cuk) if padding else None,
    )
    impl = _impl()
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), q, k, v, metadata)
    assert result.path == ("trtllm_packed_padding" if padding else "trtllm_packed_varlen")
    assert result.support.status is SupportStatus.UNMIGRATED
    assert (
        impl.resolve_execution_path(ExecutionContext(platform="cuda"), q, k, v, None).support.status
        is SupportStatus.SUPPORTED
    )


@pytest.mark.parametrize(
    "metadata,message",
    [
        (AttentionMetadata(attn_mask=torch.ones(1, 12)), "does not support attn_mask"),
        (AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 8])}), "Incomplete packed"),
        (AttentionMetadata(packed_padding="invalid"), "must be PackedPaddingMetadata"),
    ],
)
def test_resolution_and_execution_share_metadata_validation(dispatcher, metadata, message):
    impl = _impl()
    inputs = _inputs()
    with pytest.raises(ValueError, match=message):
        impl.resolve_execution_path(ExecutionContext(platform="cuda"), *inputs, metadata)
    with pytest.raises(ValueError, match=message):
        impl.forward_cuda(*inputs, metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        AttentionMetadata(full_attn_spans=[[(0, 4)]]),
        AttentionMetadata(extra={"kv_cache_dtype": "fp8"}),
    ],
)
def test_special_metadata_is_unmigrated(dispatcher, metadata):
    result = _impl().resolve_execution_path(ExecutionContext(platform="cuda"), *_inputs(), metadata)
    assert result.support.status is SupportStatus.UNMIGRATED


def test_invalid_geometry_and_missing_dependency(dispatcher, monkeypatch):
    q, k, v = _inputs()
    context = ExecutionContext(platform="cuda")
    for inputs in ((q[0], k, v), (q, k.float(), v), (q, k, v[:, :-1])):
        result = _impl().resolve_execution_path(context, *inputs, None)
        assert result.support.status is SupportStatus.UNSUPPORTED and result.support.reason
    for inputs in (
        _inputs(dtype=torch.float32),
        (q[..., :64], k[..., :64], v[..., :64]),
        (q, k[:, :, :1], v[:, :, :1]),
    ):
        assert _impl().resolve_execution_path(context, *inputs, None).support.status is SupportStatus.UNMIGRATED
    monkeypatch.setattr(tg, "HAS_FLASHINFER", False)
    result = _impl().resolve_execution_path(context, q, k, v, None)
    assert result.support.status is SupportStatus.UNSUPPORTED and "FlashInfer" in result.support.reason


@pytest.mark.parametrize("use_sage", [False, True], ids=["dense", "sage"])
def test_trtllm_dispatcher_is_opaque_to_torch_compile(dispatcher, use_sage):
    impl = _impl(backend_kwargs={"quant": {"dtype_qk": "int8"}} if use_sage else None)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True, dynamic=True)
    for q_len, kv_len in ((8, 12), (11, 17), (8, 12)):
        q, k, v = _inputs(q_len, kv_len)
        before = [t.clone() for t in (q, k, v)]
        eager = impl.forward_cuda(q, k, v)
        out = compiled(q, k, v)
        torch.testing.assert_close(out, eager)
        for actual, original in zip((q, k, v), before):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert torch.all(dispatcher == 6)


def test_custom_op_schema_fake_and_noncontiguous_output(dispatcher):
    q = torch.randn(2, 7, 128).transpose(0, 1)
    k, v = torch.randn(9, 2, 128), torch.randn(9, 2, 64)
    args = (
        q,
        k,
        v,
        dispatcher,
        torch.tensor([9], dtype=torch.int32),
        torch.tensor([0, 7], dtype=torch.int32),
        torch.tensor([0, 9], dtype=torch.int32),
        None,
        None,
        None,
        7,
        9,
        1,
        128**-0.5,
        1.0,
        -1.0,
        0,
        0,
        False,
    )
    result = torch.library.opcheck(tg._trtllm_ragged_attention_op, args, test_utils=("test_schema", "test_faketensor"))
    assert all(v == "SUCCESS" for v in result.values())
    out = tg._trtllm_ragged_attention_op(*args)
    assert out.shape == (7, 2, 64) and out.is_contiguous()


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((9, 0), False),
        ((10, 0), True),
        ((10, 3), True),
        ((10, 7), False),
        ((12, 0), False),
        (None, False),
    ],
)
def test_trtllm_device_capability_is_conservative(monkeypatch, capability, expected):
    monkeypatch.setattr(
        tg,
        "current_omni_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda device_id: capability,
        ),
    )
    assert tg._is_trtllm_blackwell_device(SimpleNamespace(device=torch.device("cuda:0"))) is expected
    assert not tg._is_trtllm_blackwell_device(SimpleNamespace(device=torch.device("cpu")))


def test_non_blackwell_cannot_claim_fullgraph(dispatcher, monkeypatch):
    monkeypatch.setattr(tg, "_is_trtllm_blackwell_device", lambda query: False)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    result = _impl().resolve_execution_path(context, *_inputs(), None)
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
