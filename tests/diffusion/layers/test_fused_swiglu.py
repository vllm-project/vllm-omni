# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import ModuleType

import pytest
import torch

from vllm_omni.diffusion.layers import fused_swiglu

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def clear_resolution_cache():
    fused_swiglu._resolve_eagle_ffn.cache_clear()
    yield
    fused_swiglu._resolve_eagle_ffn.cache_clear()


def test_non_npu_does_not_import_mindiesd(monkeypatch):
    monkeypatch.setattr(fused_swiglu.current_omni_platform, "is_npu", lambda: False)

    def unexpected_resolution():
        raise AssertionError("CPU must not resolve the optional NPU API")

    monkeypatch.setattr(fused_swiglu, "_resolve_eagle_ffn", unexpected_resolution)
    assert not fused_swiglu.fused_swiglu_available()


@pytest.mark.parametrize("available", [False, True])
def test_public_api_resolution(monkeypatch, available):
    package = ModuleType("mindiesd")
    layers = ModuleType("mindiesd.layers")

    def op(*args, **kwargs):
        return None

    if available:
        setattr(layers, "eagle_ffn_linear", op)
    setattr(package, "layers", layers)
    monkeypatch.setitem(sys.modules, "mindiesd", package)
    monkeypatch.setitem(sys.modules, "mindiesd.layers", layers)
    assert fused_swiglu._resolve_eagle_ffn() is (op if available else None)


def test_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    assert fused_swiglu._resolve_eagle_ffn() is None


def test_non_a5_does_not_resolve_operator(monkeypatch):
    from vllm_omni.platforms import npu

    monkeypatch.setattr(fused_swiglu.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(npu, "is_a5", lambda: False)

    def unexpected_resolution():
        raise AssertionError("Non-A5 NPU must keep native")

    monkeypatch.setattr(fused_swiglu, "_resolve_eagle_ffn", unexpected_resolution)
    assert not fused_swiglu.fused_swiglu_available()


def test_npu_adapter_uses_public_api_and_propagates_errors(monkeypatch):
    monkeypatch.setattr(fused_swiglu.CustomOp, "dispatch_forward", lambda op: op.forward_npu)
    layer = fused_swiglu.FusedSwiGLU()
    x, packed, down = (torch.empty(1), torch.empty(1), torch.empty(1))

    def operator(actual_x, actual_packed, actual_down, *, activation):
        assert actual_x is x and actual_packed is packed and actual_down is down
        assert activation == "swiglu"
        raise RuntimeError("operator execution failed")

    monkeypatch.setattr(fused_swiglu, "_resolve_eagle_ffn", lambda: operator)
    with pytest.raises(RuntimeError, match="operator execution failed"):
        layer(x, packed, down)
