# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for Quack compilation and warmup."""

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


def test_scale_validation_does_not_recompile_for_each_layer(monkeypatch):
    monkeypatch.setattr(quack_fp8, "_valid_scale_ptrs", set())
    torch._dynamo.reset()
    compiled_graphs = []

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    def forward(x, scale_a, scale_b):
        if quack_fp8._scales_valid(scale_a, scale_b):
            return x * (scale_a * scale_b)
        return x

    try:
        compiled = torch.compile(forward, backend=backend)
        # Keep all pairs alive, as separate model layers do. Both pointer
        # addresses and scale values differ; tensor metadata stays identical.
        with torch.inference_mode():
            pairs = [(torch.tensor([float(i + 1)]), torch.tensor([0.5])) for i in range(5)]
            x = torch.ones(4)
            for _ in range(2):
                for scale_a, scale_b in pairs:
                    torch.testing.assert_close(compiled(x, scale_a, scale_b), x * scale_a * scale_b)
        assert len(compiled_graphs) == 1
    finally:
        torch._dynamo.reset()


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
