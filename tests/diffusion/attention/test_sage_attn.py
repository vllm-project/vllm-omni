# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from dataclasses import replace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    OuterBoundary,
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
def sage_backend(monkeypatch):
    package = types.ModuleType("sageattention")
    setattr(package, "sageattn", lambda *args, **kwargs: pytest.fail("unexpected Sage kernel call"))
    monkeypatch.setitem(sys.modules, "sageattention", package)
    module_name = "vllm_omni.diffusion.attention.backends.sage_attn"
    sys.modules.pop(module_name, None)
    try:
        yield importlib.import_module(module_name), package
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("head_size", [64, 96], ids=["dense", "padded_output"])
def test_sage_attention_dispatcher_is_opaque_to_compile(sage_backend, tmp_path, head_size):
    backend, package = sage_backend
    marker = tmp_path / "kernel"
    marker.write_text("loaded")
    calls = []

    def kernel(q, k, v, *, tensor_layout, is_causal, sm_scale):
        # File I/O must not be traced, just like Sage's architecture queries
        # and pybind quantization kernels. A padded result exercises strides.
        assert marker.read_text() == "loaded"
        calls.append((tensor_layout, is_causal, sm_scale))
        output = q + k + v
        if head_size == 96:
            output = torch.nn.functional.pad(output, (0, 32))[..., :head_size]
        return output

    package.sageattn = kernel
    scale = 0.17
    impl = backend.SageAttentionImpl(4, head_size, scale, causal=False)
    q, k, v = [torch.randn(1, 12, 4, head_size) for _ in range(3)]
    expected = impl.forward_cuda(q, k, v)
    compiled = torch.compile(impl.forward_cuda, fullgraph=True)
    actual = compiled(q, k, v)
    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()
    assert calls == [("NHD", False, scale)] * 2
    if head_size == 96:
        torch.library.opcheck(backend._sage_attention_op.default, (q, k, v, False, scale))


@pytest.fixture
def sage_contract(sage_backend, monkeypatch):
    backend, _ = sage_backend
    # CPU contract tests substitute device selection only. Real SM90 dispatch
    # and fullgraph compilation are checked in test_sage_attn_compile.py.
    monkeypatch.setattr(backend, "_sage_cuda_kernel_variant", lambda query: "sage_sm90")
    return backend


def _contract_inputs(head_size=64, dtype=torch.bfloat16):
    return [torch.empty(1, 12, 4, head_size, dtype=dtype) for _ in range(3)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sage_contract_uses_actual_inputs_and_initialized_causal(sage_contract, causal, dtype):
    context = ExecutionContext(
        platform="cuda", kernel_variant="caller_invented", causal=not causal, dtype="float32", require_fullgraph=True
    )
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17, causal=causal)
    result = impl.resolve_execution_path(context, *_contract_inputs(dtype=dtype), None)
    assert result.backend == "SAGE_ATTN" and result.path == "sage_dense"
    assert result.kernel_variant == "sage_sm90"
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP
    assert result.requested_support(context).status is SupportStatus.SUPPORTED
    pre = sage_contract.SageAttentionBackend.resolve_capabilities(context)
    assert pre.support.status is SupportStatus.UNMIGRATED and pre.kernel_variant is None
    assert pre.requested_support(context).status is SupportStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "change",
    [
        {"platform": "xpu"},
        {"platform": "rocm"},
        {"paged_kv": True},
        {"parallel_strategy": ParallelStrategy.ULYSSES},
        {"parallel_strategy": ParallelStrategy.RING},
        {"parallel_strategy": ParallelStrategy.HYBRID_ULYSSES_RING},
        {"parallel_strategy": ParallelStrategy.ALLGATHER_KV},
        {"outer_boundaries": frozenset({OuterBoundary.HSDP})},
    ],
)
def test_sage_unverified_context_cannot_claim_fullgraph(sage_contract, change):
    context = replace(ExecutionContext(platform="cuda", require_fullgraph=True), **change)
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17)
    result = impl.resolve_execution_path(context, *_contract_inputs(), None)
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "metadata",
    [
        AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 12])}),
        AttentionMetadata(extra={"max_seqlen_k": 12}),
        AttentionMetadata(packed_padding=PackedPaddingMetadata(12, 12, torch.tensor([0, 12]), torch.tensor([0, 12]))),
        AttentionMetadata(full_attn_spans=[[(0, 12)]]),
        AttentionMetadata(extra={"kv_cache_dtype": "fp8"}),
    ],
)
def test_sage_special_metadata_is_not_migrated(sage_contract, metadata):
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    inputs = _contract_inputs()
    result = impl.resolve_execution_path(context, *inputs, metadata)
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
    assert impl.resolve_execution_path(context, *inputs, None).support.status is SupportStatus.SUPPORTED


@pytest.mark.parametrize("platform", ["cuda", "xpu", "rocm", "cpu"])
def test_sage_contract_and_forward_reject_masks(sage_contract, monkeypatch, platform):
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17)
    inputs = _contract_inputs()
    monkeypatch.setattr(
        sage_contract,
        "xpu_sageattn",
        lambda *args, **kwargs: pytest.fail("unexpected XPU Sage kernel call"),
        raising=False,
    )
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True] * 8 + [False] * 4]))
    context = ExecutionContext(platform=platform)
    result = impl.resolve_execution_path(context, *inputs, metadata)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert result.support.reason == "SAGE_ATTN: attn_mask is not supported"
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="does not support attn_mask"):
        impl.forward_cuda(*inputs, metadata)
    with pytest.raises(ValueError, match="does not support attn_mask"):
        impl.forward_xpu(*inputs, metadata)


def test_sage_missing_dependency(sage_contract, monkeypatch):
    monkeypatch.setattr(sage_contract, "sageattn", None)
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17)
    inputs = _contract_inputs()
    result = impl.resolve_execution_path(ExecutionContext(platform="cuda"), *inputs, None)
    assert result.support.status is SupportStatus.UNSUPPORTED and "sageattention" in result.support.reason
    with pytest.raises(ImportError, match="requires sageattention"):
        impl.forward_cuda(*inputs)


@pytest.mark.parametrize("variant", [None, "sage_sm80", "sage_sm100", "sage_sm120"])
def test_sage_architecture_is_not_taken_from_caller(sage_contract, monkeypatch, variant):
    monkeypatch.setattr(sage_contract, "_sage_cuda_kernel_variant", lambda query: variant)
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17)
    result = impl.resolve_execution_path(
        ExecutionContext(platform="cuda", kernel_variant="sage_sm90"), *_contract_inputs(), None
    )
    assert result.kernel_variant == variant and result.support.status is SupportStatus.UNMIGRATED


@pytest.mark.parametrize("variant", [None, "sage_sm80", "sage_sm90", "sage_sm120"])
def test_sage_geometry_validation(sage_contract, monkeypatch, variant):
    monkeypatch.setattr(sage_contract, "_sage_cuda_kernel_variant", lambda query: variant)
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17)
    context = ExecutionContext(platform="cuda")
    q, k, v = _contract_inputs()
    for inputs in (
        (q[0], k, v),
        (q, k.float(), v),
        (q, k, v[:, :-1]),
        (q, k, v[..., :32]),
        (q[:, :0], k[:, :0], v[:, :0]),
        (q, k.to("meta"), v),
    ):
        result = impl.resolve_execution_path(context, *inputs, None)
        assert result.support.status is SupportStatus.UNSUPPORTED and result.support.reason
    for inputs in (
        _contract_inputs(dtype=torch.float32),
        _contract_inputs(head_size=160),
        (q, k[:, :, :2], v[:, :, :2]),
        (q[..., ::2], k[..., ::2], v[..., ::2]),
    ):
        assert impl.resolve_execution_path(context, *inputs, None).support.status is SupportStatus.UNMIGRATED
    impl.causal = True
    assert impl.resolve_execution_path(context, q[:, :8], k, v, None).support.status is SupportStatus.UNMIGRATED


@pytest.mark.parametrize("platform", ["cuda", "xpu", "cpu"])
@pytest.mark.parametrize("dropout_p", [0.1, 1.0])
def test_sage_rejects_dropout_during_inspection_and_forward(sage_contract, platform, dropout_p):
    impl = sage_contract.SageAttentionImpl(4, 64, 0.17, dropout_p=dropout_p)
    inputs = _contract_inputs()
    context = ExecutionContext(platform=platform)
    result = impl.resolve_execution_path(context, *inputs, None)
    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "does not support dropout" in result.support.reason
    for forward in (impl.forward_cuda, impl.forward_xpu):
        with pytest.raises(ValueError, match="does not support dropout"):
            forward(*inputs)
