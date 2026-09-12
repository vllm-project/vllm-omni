# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers import swiglu7 as kernel_module
from vllm_omni.diffusion.layers.swiglu7 import SwiGLU7
from vllm_omni.diffusion.models.magi2 import layers as magi2_layers

pytestmark = [pytest.mark.diffusion, pytest.mark.local_model]


def reference(x, alpha=1.702, limit=7.0, out_dtype=None):
    gate = x[..., ::2].float().clamp(max=limit)
    up = x[..., 1::2].float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + 1)).to(x.dtype if out_dtype is None else out_dtype)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("out_dtype", [None, torch.float32, torch.bfloat16])
def test_native_dtype_and_rounding(dtype, out_dtype):
    torch.manual_seed(71)
    x = torch.randn(3, 14).to(dtype)
    actual = SwiGLU7().forward_native(x, out_dtype=out_dtype)
    expected = reference(x, out_dtype=out_dtype)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("mode", ["noncontiguous", "empty", "rank3", "odd_broadcast", "odd_error", "custom"])
def test_native_fallback_preserves_existing_inputs(mode):
    op = SwiGLU7()
    x = torch.randn(3, 12)
    kwargs = {}
    if mode == "noncontiguous":
        x = x[:, ::2]
    elif mode == "empty":
        x = torch.empty(0, 12)
    elif mode == "rank3":
        x = torch.randn(2, 3, 12)
    elif mode == "odd_broadcast":
        x = torch.randn(3, 3)
    elif mode == "odd_error":
        with pytest.raises(RuntimeError):
            op.forward_cuda(torch.randn(3, 7))
        return
    elif mode == "custom":
        kwargs = dict(alpha=0.5, limit=2.0, out_dtype=torch.float16)
    torch.testing.assert_close(op.forward_cuda(x, **kwargs), reference(x, **kwargs), rtol=0, atol=0)


@pytest.mark.cpu
def test_meta_output_shape():
    x = torch.empty(2, 5, 18, device="meta", dtype=torch.bfloat16)
    out = SwiGLU7().forward_cuda(x, out_dtype=torch.float32)
    assert out.shape == (2, 5, 9) and out.dtype == torch.float32 and out.device.type == "meta"


@pytest.mark.cpu
def test_gradients_remain_native():
    x = torch.randn(3, 12, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = SwiGLU7().forward_cuda(x)
    expected = reference(y)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)


@pytest.mark.cpu
def test_model_wiring_is_opt_in(monkeypatch):
    x = torch.randn(3, 12)
    calls = []

    class Probe:
        def __call__(self, x, alpha, limit, out_dtype):
            calls.append("dispatch")
            return reference(x, alpha, limit, out_dtype)

        def forward_native(self, x, alpha, limit, out_dtype):
            calls.append("native")
            return reference(x, alpha, limit, out_dtype)

    monkeypatch.setattr(magi2_layers, "_swiglu7_op", Probe())
    monkeypatch.delenv("MAGI2_USE_FUSED_SWIGLU7", raising=False)
    torch.testing.assert_close(magi2_layers.swiglu7(x), reference(x))
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "1")
    torch.testing.assert_close(magi2_layers.swiglu7(x), reference(x))
    assert calls == ["native", "dispatch"]


@pytest.mark.cpu
def test_compile_keeps_native_pointwise_graph(monkeypatch):
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "1")
    torch._dynamo.reset()
    targets: list[str] = []

    def backend(gm, _inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes)
        return gm.forward

    compiled = torch.compile(magi2_layers.swiglu7, backend=backend, fullgraph=True)
    x = torch.randn(3, 12)
    torch.testing.assert_close(compiled(x), reference(x), rtol=0, atol=0)
    assert any("sigmoid" in target for target in targets)
    assert not any("triton" in target or "swiglu7_kernel" in target for target in targets)


GPU_DEVICES = [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("musa", marks=pytest.mark.musa)]


def require_device(name):
    backend = getattr(torch, name, None)
    if backend is None or not backend.is_available():
        pytest.skip(f"requires {name}")
    return torch.device(name)


@pytest.mark.parametrize("device_name", GPU_DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(129, 2560), (2, 514), (2, 3, 128), (0, 2560)])
def test_eager_kernel_launch_and_numerics(monkeypatch, device_name, dtype, shape):
    device = require_device(device_name)
    torch.manual_seed(42)
    x = (torch.randn(shape, dtype=torch.float32, device=device) * 10).to(dtype)
    before = x.clone()
    real_kernel = kernel_module._swiglu7_kernel
    launches = []

    class Spy:
        def __getitem__(self, grid):
            launches.append(grid)
            return real_kernel[grid]

    monkeypatch.setattr(kernel_module, "_swiglu7_kernel", Spy())
    actual = SwiGLU7()(x)
    getattr(torch, device_name).synchronize()
    assert len(launches) == (1 if x.numel() else 0)
    # Up to two output ULPs accommodate FP32 sigmoid implementation differences
    # before the final BF16/FP16 rounding; FP32 keeps a tighter numerical gate.
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 2e-3, torch.float32: 1e-5}[dtype]
    atol = 1e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(actual, reference(x), rtol=rtol, atol=atol)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", GPU_DEVICES)
def test_special_values_and_explicit_output_dtype(device_name):
    device = require_device(device_name)
    x = torch.tensor(
        [
            [
                -100.0,
                2.0,
                -8.0,
                1.0,
                -7.0,
                0.0,
                0.0,
                -1.0,
                8.0,
                100.0,
                float("inf"),
                -100.0,
                float("-inf"),
                0.0,
                float("nan"),
                2.0,
                1.0,
                float("nan"),
            ]
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    actual = SwiGLU7()(x, out_dtype=torch.float32)
    expected = reference(x, out_dtype=torch.float32)
    getattr(torch, device_name).synchronize()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("device_name", GPU_DEVICES)
def test_gpu_compile_does_not_launch_standalone_kernel(monkeypatch, device_name):
    device = require_device(device_name)

    class ForbiddenKernel:
        def __getitem__(self, _grid):
            pytest.fail("compilation must use the visible native expression")

    monkeypatch.setattr(kernel_module, "_swiglu7_kernel", ForbiddenKernel())
    op = torch.compile(SwiGLU7(), backend="eager", fullgraph=True)
    x = torch.randn(7, 128, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(op(x), reference(x), rtol=0, atol=0)
