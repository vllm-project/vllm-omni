# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native fallback must retain GroupNorm numerics and autograd on CPU."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def adapter():
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_omni/diffusion/models/hunyuan_image3/layers/npu/normalization.py"
    spec = importlib.util.spec_from_file_location("vae_distgn_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.vae_group_norm_silu


@pytest.mark.parametrize("shape", [(2, 128, 3, 4, 5), (1, 96, 4, 5), (1, 128, 0, 4, 5)])
def test_native_fallback_output_and_gradient(adapter, shape):
    x = torch.randn(shape, requires_grad=True).transpose(-1, -2)
    norm = torch.nn.GroupNorm(32, shape[1], eps=1e-6)
    actual = adapter(x, norm)
    expected = torch.nn.functional.silu(norm(x))
    assert torch.equal(actual, expected)
    if actual.numel():
        grad_actual = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
        grad_expected = torch.autograd.grad(expected.sum(), x)[0]
        assert torch.equal(grad_actual, grad_expected)


@pytest.mark.parametrize("activation", ["swish", "identity"])
def test_mixed_conv_path_preserves_cpu_autograd(adapter, activation):
    fused = adapter.__globals__["vae_group_norm_for_conv"]
    x = torch.randn(1, 128, 2, 3, 4, requires_grad=True).transpose(-1, -2)
    norm = torch.nn.GroupNorm(32, 128, eps=1e-6)
    expected = norm(x)
    if activation == "swish":
        expected = torch.nn.functional.silu(expected)
    actual = fused(x, norm, activation)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), x)[0],
    )


def test_mixed_conv_path_rejects_unknown_activation(adapter):
    fused = adapter.__globals__["vae_group_norm_for_conv"]
    with pytest.raises(ValueError, match="activation"):
        fused(torch.randn(1, 128, 2, 3), torch.nn.GroupNorm(32, 128), "relu")


@pytest.mark.parametrize("distributed", [False, True])
def test_explicit_ncthw_layout_dispatch(monkeypatch, distributed):
    from vllm_omni.platforms.npu.layers import dist_group_norm as cf
    from vllm_omni.platforms.npu.layers import dist_group_norm as op

    x = torch.randn(1, 32, 1, 2, 2)
    group = object()
    calls = []

    def kernel(actual, groups, weight, bias, eps, *args, **kwargs):
        calls.append((actual, weight, bias, args, kwargs))
        return actual

    monkeypatch.setattr(op.dist, "is_initialized", lambda: distributed)
    monkeypatch.setattr(cf, "_group_norm_ncthw", kernel)
    monkeypatch.setattr(cf, "_distributed_group_norm_ncthw", kernel)
    assert op.dist_group_norm(x, 32, None, None, 1e-6, device_group=group, layout="NCTHW") is x
    assert len(calls) == 1
    assert torch.equal(calls[0][1], torch.ones(32))
    assert torch.equal(calls[0][2], torch.zeros(32))
    assert calls[0][3] == ((group,) if distributed else ())


def test_explicit_nhwc_layout_dispatch(monkeypatch):
    from vllm_omni.platforms.npu.layers import dist_group_norm as op

    x = torch.randn(1, 2, 2, 32)
    monkeypatch.setattr(op, "_collect", lambda actual, groups: (actual, None, None))
    monkeypatch.setattr(op, "_reduce", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(op, "_apply", lambda actual, *args: actual)
    assert op.dist_group_norm(x, 32, None, None, 1e-6, layout="NHWC", local_only=True) is x


@pytest.mark.parametrize("layout,shape", [("NCHW", (1, 32, 2, 2)), ("NCTHW", (1, 32, 2, 2))])
def test_layout_contract_rejects_unsupported_input(layout, shape):
    from vllm_omni.platforms.npu.layers.dist_group_norm import dist_group_norm

    with pytest.raises(ValueError):
        dist_group_norm(torch.randn(shape), 32, None, None, 1e-6, layout=layout)


@pytest.mark.parametrize("activation", ["swish", "identity"])
def test_independent_tiles_keep_native_norm_even_for_large_npu_shapes(adapter, activation):
    from types import SimpleNamespace

    x = SimpleNamespace(
        device=SimpleNamespace(type="npu"),
        dtype=torch.float32,
        ndim=5,
        is_contiguous=lambda: True,
        numel=lambda: 16 * 1024 * 1024,
    )
    expected = torch.randn(1, 32, 1, 2, 2)
    calls = []

    class NativeNorm:
        num_groups = 32
        weight = torch.ones(32)
        bias = torch.zeros(32)
        eps = 1e-6

        def __call__(self, actual):
            assert actual is x
            calls.append(actual)
            return expected

    with torch.no_grad():
        actual = adapter.__globals__["vae_group_norm_for_conv"](x, NativeNorm(), activation)
    reference = torch.nn.functional.silu(expected) if activation == "swish" else expected
    assert torch.equal(actual, reference)
    assert len(calls) == 1
