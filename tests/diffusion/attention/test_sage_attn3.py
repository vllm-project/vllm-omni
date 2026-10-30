# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import sys
import types

import pytest
import torch
from vllm.platforms import current_platform

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


SAGE_ATTN3_MODULE = "vllm_omni.diffusion.attention.backends.sage_attn3"


def load_sage_attn3_module(monkeypatch: pytest.MonkeyPatch, kernel_impl):
    fake_module = types.ModuleType("sageattn3")
    setattr(fake_module, "sageattn3_blackwell", kernel_impl)
    monkeypatch.setitem(sys.modules, "sageattn3", fake_module)
    monkeypatch.delitem(sys.modules, SAGE_ATTN3_MODULE, raising=False)
    return importlib.import_module(SAGE_ATTN3_MODULE)


def test_sage_attn3_forward_uses_blackwell_layout(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_kernel(query, key, value, is_causal=False):
        calls["query_shape"] = query.shape
        calls["is_causal"] = is_causal
        return query + key + value

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    impl = backend_module.SageAttention3Impl(
        num_heads=4,
        head_size=64,
        softmax_scale=1.0 / 8.0,
        causal=False,
    )

    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 4, 64)
    value = torch.randn(2, 8, 4, 64)

    output = impl.forward_cuda(query, key, value)

    assert calls["query_shape"] == (2, 4, 8, 64)
    assert calls["is_causal"] is False
    expected = (query.transpose(1, 2) + key.transpose(1, 2) + value.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(output, expected)


def test_sage_attn3_rejects_custom_softmax_scale(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module(monkeypatch, lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="does not expose a custom softmax scale"):
        backend_module.SageAttention3Impl(
            num_heads=4,
            head_size=64,
            softmax_scale=1.0,
            causal=False,
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
def test_cuda_platform_selects_sage_attn3_alias(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda import platform as cuda_platform_module
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    original_import_module = importlib.import_module

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(
        cuda_platform_module.importlib,
        "import_module",
        lambda module_name: object() if module_name == "sageattn3" else original_import_module(module_name),
    )

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)

    assert backend_path == DiffusionAttentionBackendEnum.SAGE_ATTN_3.get_path()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
@pytest.mark.parametrize("head_size", [32, 320])
def test_cuda_platform_rejects_explicit_sage_attn3_unsupported_head_size(
    monkeypatch: pytest.MonkeyPatch, head_size: int
):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda import platform as cuda_platform_module
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    original_import_module = importlib.import_module

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(
        cuda_platform_module.importlib,
        "import_module",
        lambda module_name: object() if module_name == "sageattn3" else original_import_module(module_name),
    )
    load_sage_attn3_module(monkeypatch, lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match=f"head_size={head_size} is unsupported"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=head_size)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
def test_cuda_platform_rejects_explicit_sage_attn3_on_unsupported_gpu(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})

    with pytest.raises(ValueError, match="explicitly selected"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
def test_cuda_platform_rejects_missing_explicit_sage_attn3(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda import platform as cuda_platform_module
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    original_import_module = importlib.import_module

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})

    def _missing_sageattn3(module_name):
        if module_name == "sageattn3":
            raise ImportError("sageattn3 not installed")
        return original_import_module(module_name)

    monkeypatch.setattr(cuda_platform_module.importlib, "import_module", _missing_sageattn3)

    with pytest.raises(ImportError, match="explicitly selected"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)


@pytest.mark.parametrize("length", [1, 17])
def test_sage3_custom_op_owns_mutated_key(monkeypatch, length):
    def kernel(q, k, v, is_causal=False):
        k.sub_(k.mean(dim=-2, keepdim=True))
        # Exercise a noncontiguous vendor output as well as in-place K centering.
        return (q + k + v).transpose(-1, -2).contiguous().transpose(-1, -2)

    module = load_sage_attn3_module(monkeypatch, kernel)
    q = torch.randn(1, 1, length, 64)
    op = module._sageattn3_blackwell_op.default
    checks = torch.library.opcheck(op, (q, q, q, False), test_utils=("test_schema", "test_faketensor"))
    assert all(result == "SUCCESS" for result in checks.values())
    # The custom op takes HND; the forward method takes NHD.
    q = q.transpose(1, 2)
    original = q.clone()
    impl = module.SageAttention3Impl(1, 64, 0.125)
    torch.compiler.reset()
    try:
        compiled = torch.compile(impl.forward_cuda, fullgraph=True, dynamic=True)
        for _ in range(2):
            out = compiled(q, q, q)
            torch.testing.assert_close(out, impl.forward_cuda(q, q, q))
            assert out.is_contiguous()
            torch.testing.assert_close(q, original, atol=0, rtol=0)
    finally:
        torch.compiler.reset()


def test_sage3_contract_does_not_inherit_sage2_verification(monkeypatch):
    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

    module = load_sage_attn3_module(monkeypatch, lambda q, k, v, **kw: q + k + v)
    context = ExecutionContext(platform="cuda", kernel_variant="sage_sm90", require_fullgraph=True)
    impl = module.SageAttention3Impl(1, 64, 0.125)
    q = torch.randn(1, 17, 1, 64)
    for result in (
        module.SageAttention3Backend.resolve_capabilities(context),
        impl.resolve_execution_path(context, q, q, q, None),
    ):
        assert result.backend == "SAGE_ATTN_3"
        assert result.kernel_variant is None
        assert result.support.status is SupportStatus.UNMIGRATED
        assert result.requested_support(context).status is not SupportStatus.SUPPORTED


@pytest.mark.parametrize("variant", ["sage3_sm90", "sage3_sm100", "sage3_sm103", "sage3_sm120", "sage3_sm121"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128])
def test_sage3_verified_device_scope(monkeypatch, variant, dtype, head_size):
    from vllm_omni.diffusion.attention.capabilities import CompilationMode, ExecutionContext, SupportStatus

    module = load_sage_attn3_module(monkeypatch, lambda q, k, v, **kw: q + k + v)
    monkeypatch.setattr(module, "_sage3_kernel_variant", lambda q: variant)
    impl = module.SageAttention3Impl(2, head_size, head_size**-0.5)
    q = torch.randn(1, 17, 2, head_size, dtype=dtype)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    result = impl.resolve_execution_path(context, q, q, q, None)
    assert result.kernel_variant == variant
    expected = SupportStatus.SUPPORTED if variant == "sage3_sm120" else SupportStatus.UNMIGRATED
    assert result.support.status is expected
    if expected is SupportStatus.SUPPORTED:
        assert result.compilation_mode is CompilationMode.CUSTOM_OP
        assert result.requested_support(context).status is SupportStatus.SUPPORTED


@pytest.mark.parametrize(
    "case",
    [
        "packed",
        "piecewise",
        "paged",
        "parallel",
        "hsdp",
        "kv_quant",
        "head256",
        "causal_cross",
        "dtype",
        "shape",
    ],
)
def test_sage3_unverified_and_invalid_paths(monkeypatch, case):
    from vllm_omni.diffusion.attention.capabilities import (
        ExecutionContext,
        OuterBoundary,
        ParallelStrategy,
        SupportStatus,
    )

    module = load_sage_attn3_module(monkeypatch, lambda q, k, v, **kw: q + k + v)
    monkeypatch.setattr(module, "_sage3_kernel_variant", lambda q: "sage3_sm120")
    head = 256 if case == "head256" else 64
    impl = module.SageAttention3Impl(2, head, head**-0.5, causal=case == "causal_cross")
    q, k, v = (torch.randn(1, 17, 2, head, dtype=torch.bfloat16) for _ in range(3))
    context = ExecutionContext(
        platform="cuda",
        paged_kv=case == "paged",
        parallel_strategy=ParallelStrategy.ULYSSES if case == "parallel" else ParallelStrategy.NONE,
        outer_boundaries=frozenset({OuterBoundary.HSDP}) if case == "hsdp" else frozenset(),
    )
    metadata = None
    if case == "packed":
        metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 17])})
    elif case == "piecewise":
        metadata = AttentionMetadata(full_attn_spans=[[(0, 17)]])
    elif case == "kv_quant":
        metadata = AttentionMetadata(extra={"kv_cache_dtype": "fp8"})
    elif case == "causal_cross":
        k, v = k[:, :12], v[:, :12]
    elif case == "dtype":
        k = k.float()
    elif case == "shape":
        v = v[:, :12]
    result = impl.resolve_execution_path(context, q, k, v, metadata)
    expected = SupportStatus.UNSUPPORTED if case in ("dtype", "shape") else SupportStatus.UNMIGRATED
    assert result.support.status is expected


@pytest.mark.parametrize("variant", [None, "sage3_sm90", "sage3_sm100", "sage3_sm120", "sage3_sm121"])
@pytest.mark.parametrize("kv_heads", [1, 2, 3])
def test_sage3_capabilities_reject_head_mismatch_before_variant_gate(monkeypatch, variant, kv_heads):
    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

    def kernel(*args, **kwargs):
        raise AssertionError("Capability resolution must not invoke the kernel")

    module = load_sage_attn3_module(monkeypatch, kernel)
    monkeypatch.setattr(module, "_sage3_kernel_variant", lambda q: variant)
    impl = module.SageAttention3Impl(4, 64, 0.125)
    q = torch.empty(1, 17, 4, 64, dtype=torch.bfloat16)
    k = v = torch.empty(1, 17, kv_heads, 64, dtype=torch.bfloat16)
    context = ExecutionContext(platform="cuda" if variant else "cpu")

    result = impl.resolve_execution_path(context, q, k, v, None)

    assert result.support.status is SupportStatus.UNSUPPORTED
    assert "does not support GQA/MQA" in result.support.reason
    assert f"q_heads=4, kv_heads={kv_heads}" in result.support.reason
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
    error = ValueError if kv_heads == 3 else NotImplementedError
    with pytest.raises(error, match="GQA/MQA"):
        impl.forward_cuda(q, k, v)


@pytest.mark.parametrize("variant", [None, "sage3_sm100", "sage3_sm120", "sage3_sm121"])
@pytest.mark.parametrize("head_size,runtime_head_size", [(64, 128), (128, 64), (64, 256)])
def test_sage3_rejects_runtime_scale_mismatch(monkeypatch, variant, head_size, runtime_head_size):
    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

    def kernel(*args, **kwargs):
        raise AssertionError("Sage3 must reject a scale mismatch before invoking the kernel")

    module = load_sage_attn3_module(monkeypatch, kernel)
    monkeypatch.setattr(module, "_sage3_kernel_variant", lambda q: variant)
    scale = head_size**-0.5
    impl = module.SageAttention3Impl(2, head_size, scale)
    q = torch.empty(1, 17, 2, runtime_head_size, dtype=torch.bfloat16)
    context = ExecutionContext(platform="cuda" if variant else "cpu")
    reason = f"SAGE_ATTN_3: softmax_scale {scale} does not match head dim {runtime_head_size}."

    result = impl.resolve_execution_path(context, q, q, q, None)

    assert result.support.status is SupportStatus.UNSUPPORTED
    assert result.support.reason == reason
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError) as exc_info:
        impl.forward_cuda(q, q, q)
    assert str(exc_info.value) == reason


@pytest.mark.parametrize("variant", [None, "sage3_sm100", "sage3_sm120", "sage3_sm121"])
@pytest.mark.parametrize("dropout_p", [0.1, 1.0])
def test_sage3_rejects_dropout_before_variant_gate_and_kernel(monkeypatch, variant, dropout_p):
    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

    def kernel(*args, **kwargs):
        raise AssertionError("Sage3 must reject dropout before invoking the kernel")

    module = load_sage_attn3_module(monkeypatch, kernel)
    monkeypatch.setattr(module, "_sage3_kernel_variant", lambda q: variant)
    impl = module.SageAttention3Impl(2, 64, 0.125, dropout_p=dropout_p)
    q = torch.empty(1, 17, 2, 64, dtype=torch.bfloat16)
    context = ExecutionContext(platform="cuda" if variant else "cpu")
    reason = f"SAGE_ATTN_3: does not support dropout (dropout_p={dropout_p})."

    result = impl.resolve_execution_path(context, q, q, q, None)

    assert result.support.status is SupportStatus.UNSUPPORTED
    assert result.support.reason == reason
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError) as exc_info:
        impl.forward_cuda(q, q, q)
    assert str(exc_info.value) == reason


@pytest.mark.parametrize("variant", [None, "sage3_sm100", "sage3_sm120", "sage3_sm121"])
def test_sage3_mask_inspection_returns_unsupported(monkeypatch, variant):
    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus

    def kernel(*args, **kwargs):
        raise AssertionError("Sage3 must reject masks before invoking the kernel")

    module = load_sage_attn3_module(monkeypatch, kernel)
    monkeypatch.setattr(module, "_sage3_kernel_variant", lambda q: variant)
    impl = module.SageAttention3Impl(2, 64, 0.125)
    q = torch.empty(1, 12, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True] * 8 + [False] * 4]))
    context = ExecutionContext(platform="cuda" if variant else "cpu")

    result = impl.resolve_execution_path(context, q, q, q, metadata)

    assert result.support.status is SupportStatus.UNSUPPORTED
    assert result.support.reason == "SAGE_ATTN_3: attn_mask is not supported"
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="does not support attn_mask"):
        impl.forward_cuda(q, q, q, metadata)
