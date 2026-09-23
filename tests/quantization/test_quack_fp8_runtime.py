# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for Quack compilation and warmup."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.quantization import quack_fp8

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_compile_pool_checks_daemon_at_tuning_time(monkeypatch):
    process = SimpleNamespace(daemon=False)
    compiler_pool = object()
    pool = SimpleNamespace(
        pool_scope=Mock(spec=[], side_effect=lambda: nullcontext(compiler_pool)),
        suppress_pool=Mock(side_effect=lambda: nullcontext(object())),
    )
    original_scope = pool.pool_scope
    monkeypatch.setattr(quack_fp8, "current_process", lambda: process)
    monkeypatch.setattr(quack_fp8, "import_module", lambda name: pool)
    quack_fp8._configure_quack_compilation()
    with pool.pool_scope() as active_pool:
        assert active_pool is compiler_pool
    pool.suppress_pool.assert_not_called()

    # spawn installs the child's daemon flag after importing the worker module.
    process.daemon = True
    with pool.pool_scope() as active_pool:
        assert active_pool is None
    original_scope.assert_called_once_with()
    pool.suppress_pool.assert_called_once_with()


@pytest.fixture
def quack_dispatch(monkeypatch):
    """Exercise the installed patch with CPU substitutes for the GPU kernels."""

    def gemm(a, b, *, out, bias, alpha, tuned):
        assert isinstance(alpha, torch.Tensor)
        assert alpha.shape == (1,)
        assert alpha.dtype == torch.float32
        assert tuned is True
        result = (a.float() @ b.float()) * alpha
        if bias is not None:
            result += bias
        out.copy_(result)

    def flashinfer(a, b, *, out_dtype, scale_a, scale_b, bias):
        out = torch.full((a.shape[0], b.shape[1]), -7, dtype=out_dtype, device=a.device)
        return out + bias if bias is not None else out

    class FlashInferKernel:
        def apply_scaled_mm(self, **kwargs):
            raise AssertionError("The installed custom op should handle dispatch")

    gemm_mock = Mock(side_effect=gemm)
    fallback_mock = Mock(side_effect=flashinfer)
    monkeypatch.setattr(quack_fp8, "quack_enabled", lambda: True)
    monkeypatch.setattr(quack_fp8, "_gemm_interface", SimpleNamespace(gemm=gemm_mock))
    monkeypatch.setattr(quack_fp8, "_valid_scale_ptrs", set())
    monkeypatch.setattr(quack_fp8, "logger", Mock())
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.linear.scaled_mm.flashinfer",
        SimpleNamespace(FlashInferFP8ScaledMMLinearKernel=FlashInferKernel),
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.flashinfer", SimpleNamespace(flashinfer_scaled_fp8_mm=fallback_mock))
    quack_fp8.install_quack_fp8_patch()
    kernel = FlashInferKernel()

    def forward(a, b, scale_a, scale_b, bias=None, out_dtype=torch.bfloat16):
        return kernel.apply_scaled_mm(
            A=a,
            B=b,
            As=scale_a,
            Bs=scale_b,
            out_dtype=out_dtype,
            bias=bias,
            output_shape=(2, a.shape[0] // 2, b.shape[1]),
        )

    return SimpleNamespace(forward=forward, gemm=gemm_mock, fallback=fallback_mock)


@pytest.fixture
def compile_fullgraph():
    torch._dynamo.reset()
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    def compile_fn(fn, **kwargs):
        return torch.compile(fn, backend=backend, fullgraph=True, **kwargs)

    try:
        yield compile_fn, graphs
    finally:
        torch._dynamo.reset()


def _fp8_inputs(m=4):
    a = torch.arange(m * 8, dtype=torch.float32).reshape(m, 8).to(torch.float8_e4m3fn)
    b = torch.ones(6, 8, dtype=torch.float8_e4m3fn).t()
    return a, b


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_scale_validation_does_not_recompile_for_each_layer(quack_dispatch, compile_fullgraph, out_dtype, with_bias):
    compile_fn, graphs = compile_fullgraph
    compiled = compile_fn(quack_dispatch.forward)
    with torch.inference_mode():
        a, b = _fp8_inputs()
        bias = torch.arange(6, dtype=out_dtype) if with_bias else None
        # Keep all pairs alive, as separate model layers do. Both pointer
        # addresses and scale values differ; tensor metadata stays identical.
        pairs = [(torch.tensor([float(i + 1)]), torch.tensor([0.5])) for i in range(5)]
        for _ in range(2):
            for scale_a, scale_b in pairs:
                expected = (a.float() @ b.float()) * scale_a * scale_b
                if bias is not None:
                    expected += bias
                out = compiled(a, b, scale_a, scale_b, bias, out_dtype)
                torch.testing.assert_close(out, expected.to(out_dtype).view(2, 2, 6))
    assert len(graphs) == 1
    assert any(node.target == torch.ops.vllm_omni.quack_fp8_scaled_mm.default for node in graphs[0].graph.nodes)
    assert quack_dispatch.gemm.call_count == 10
    quack_dispatch.fallback.assert_not_called()


def test_compiled_dispatch_rechecks_invalid_scales_after_loading(quack_dispatch, compile_fullgraph):
    compile_fn, graphs = compile_fullgraph
    compiled = compile_fn(quack_dispatch.forward)
    with torch.inference_mode():
        a, b = _fp8_inputs()
        pairs = []
        for bad_scale in (torch.finfo(torch.float32).min, 0.0, -1.0, float("nan"), float("inf")):
            for scale_index in (0, 1):
                scales = [torch.ones(1), torch.ones(1)]
                scales[scale_index].fill_(bad_scale)
                pairs.append(scales)
        for scale_a, scale_b in pairs:
            out = compiled(a, b, scale_a, scale_b)
            torch.testing.assert_close(out, torch.full((2, 2, 6), -7, dtype=torch.bfloat16))
        quack_dispatch.gemm.assert_not_called()
        assert quack_dispatch.fallback.call_count == len(pairs)

        # Populate the same buffers and switch branches without recompiling.
        for scale_a, scale_b in pairs:
            scale_a.fill_(0.25)
            scale_b.fill_(0.5)
            out = compiled(a, b, scale_a, scale_b)
            expected = ((a.float() @ b.float()) * 0.125).to(torch.bfloat16)
            torch.testing.assert_close(out, expected.view(2, 2, 6))
        assert quack_dispatch.gemm.call_count == len(pairs)
        assert quack_dispatch.fallback.call_count == len(pairs)
    assert len(graphs) == 1


def test_compiled_dispatch_falls_back_on_quack_failure(monkeypatch, quack_dispatch, compile_fullgraph):
    compile_fn, graphs = compile_fullgraph
    compiled = compile_fn(quack_dispatch.forward)
    with torch.inference_mode():
        a, b = _fp8_inputs()
        scale_a, scale_b = torch.ones(1), torch.ones(1)
        bias = torch.arange(6, dtype=torch.float32)
        expected = (torch.full((2, 2, 6), -7, dtype=torch.bfloat16) + bias).to(torch.bfloat16)
        compiled(a, b, scale_a, scale_b, bias)
        quack_dispatch.fallback.assert_not_called()

        quack_dispatch.gemm.side_effect = RuntimeError("kernel failed")
        out = compiled(a, b, scale_a, scale_b, bias)
        torch.testing.assert_close(out, expected)
        quack_fp8.logger.warning_once.assert_called_once()

        # The helper can also return None when Quack is unavailable.
        monkeypatch.setattr(quack_fp8, "_gemm_interface", False)
        out = compiled(a, b, scale_a, scale_b, bias)
        torch.testing.assert_close(out, expected)
        assert quack_dispatch.fallback.call_count == 2
    assert len(graphs) == 1


@pytest.mark.parametrize("scale_index", [0, 1])
def test_compiled_dispatch_falls_back_for_non_scalar_scales(quack_dispatch, compile_fullgraph, scale_index):
    compile_fn, graphs = compile_fullgraph
    compiled = compile_fn(quack_dispatch.forward)
    with torch.inference_mode():
        a, b = _fp8_inputs()
        scales = [torch.ones(1), torch.ones(1)]
        scales[scale_index] = torch.ones(2)
        out = compiled(a, b, *scales)
        torch.testing.assert_close(out, torch.full((2, 2, 6), -7, dtype=torch.bfloat16))
    quack_dispatch.gemm.assert_not_called()
    quack_dispatch.fallback.assert_called_once()
    assert len(graphs) == 1


@pytest.mark.parametrize("valid_scales", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_dispatch_custom_op_contract(quack_dispatch, valid_scales, with_bias):
    a, b = _fp8_inputs()
    # opcheck's CPU mutation checker uses allclose, which does not support FP8.
    # The compile tests above exercise real FP8 inputs with these same mocks.
    a, b = a.float(), b.float()
    scale_a = torch.tensor([0.25 if valid_scales else torch.finfo(torch.float32).min])
    scale_b = torch.tensor([0.5])
    bias = torch.arange(6, dtype=torch.float32) if with_bias else None
    torch.library.opcheck(quack_fp8._quack_fp8_scaled_mm, (a, b, scale_a, scale_b, torch.bfloat16, bias))


def test_warmup_matches_inference_weight_layout(monkeypatch):
    calls = []

    def gemm(a, b, *, out, bias, alpha, tuned):
        assert torch.is_inference_mode_enabled()
        assert a.dtype == b.dtype == torch.float8_e4m3fn
        assert a.is_contiguous()
        assert b.stride() == (1, b.shape[0])
        assert bias is None
        assert tuned is True
        calls.append((a.shape[0], a.shape[1], b.shape[1]))
        out.zero_()

    monkeypatch.setattr(quack_fp8, "_gemm_interface", SimpleNamespace(gemm=gemm))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    shapes = [(4, 8, 6), (2, 16, 8)]
    quack_fp8.warmup_quack_fp8(shapes, device="cpu")
    assert calls == shapes
