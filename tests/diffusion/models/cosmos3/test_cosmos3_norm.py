# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU reference and mocked-NPU dispatch contracts; no NPU numerics claim."""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers import custom_op
from vllm_omni.diffusion.layers import norm as shared_norm
from vllm_omni.diffusion.models.cosmos3.norm import QKRMSNorm, RMSNorm, VAERMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("method", ["forward_native", "forward_cuda", "forward_hip", "forward_xpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_qk_norm_preserves_functional_formula(method: str, dtype: torch.dtype) -> None:
    norm = QKRMSNorm(4, eps=1e-5, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, -0.75, 1.25, 2.0], dtype=dtype))
    # Noncontiguous BSND input with GQA-sized head axes handled independently.
    q = torch.linspace(-2, 3, 48).reshape(2, 2, 3, 4).transpose(1, 2).to(dtype)
    k = q[:, :, :1, :]
    for x in (q, k):
        before, weight = x.clone(), norm.weight.detach().clone()
        expected = F.rms_norm(x, (4,), norm.weight, eps=1e-5)
        actual = getattr(norm, method)(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        torch.testing.assert_close(norm.weight, weight, rtol=0, atol=0)
        assert actual.dtype == dtype


@pytest.mark.parametrize("method", ["forward_native", "forward_cuda", "forward_hip", "forward_xpu"])
def test_qk_native_paths_reject_residual(method: str) -> None:
    x = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match="does not support residual"):
        getattr(QKRMSNorm(4), method)(x, x.clone())


@pytest.mark.parametrize("method", ["forward_native", "forward_cuda", "forward_hip", "forward_xpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_hidden_norm_preserves_fp32_formula_and_residual(method: str, dtype: torch.dtype, with_residual: bool) -> None:
    norm = RMSNorm(4, eps=1e-5, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, -0.75, 1.25, 2.0], dtype=dtype))
    x = torch.linspace(-2, 3, 24).reshape(2, 3, 4).to(dtype)
    residual = torch.flip(x, dims=[-1]) * 0.25 if with_residual else None
    before, weight = x.clone(), norm.weight.detach().clone()
    residual_before = residual.clone() if residual is not None else None
    summed = residual + x if residual is not None else x
    fp32 = summed.float()
    expected = (norm.weight.float() * (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-5))).to(dtype)
    result = getattr(norm, method)(x, residual)
    if residual is not None:
        actual, updated_residual = result
        torch.testing.assert_close(updated_residual, summed, rtol=0, atol=0)
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)
    else:
        actual = result
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(norm.weight, weight, rtol=0, atol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("norm_cls,with_residual", [(RMSNorm, False), (RMSNorm, True), (QKRMSNorm, False)])
def test_npu_dispatch_uses_shared_operator_contract(monkeypatch, norm_cls, with_residual: bool) -> None:
    """Mock device detection/operators, not the model's dispatch or forward_npu."""
    platform = custom_op.current_omni_platform
    monkeypatch.setattr(platform, "is_rocm", lambda: False)
    monkeypatch.setattr(platform, "is_cuda", lambda: False)
    monkeypatch.setattr(platform, "is_npu", lambda: True)
    monkeypatch.setattr(shared_norm, "_is_npu_tensor", lambda x: True)
    norm = norm_cls(4, eps=1e-5)
    assert norm._forward_method.__func__ is shared_norm.RMSNorm.forward_npu
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 8
    residual = torch.ones_like(x) if with_residual else None
    before, weight = x.clone(), norm.weight.detach().clone()
    residual_before = residual.clone() if residual is not None else None
    output, updated = torch.full_like(x, -3), torch.full_like(x, 7)
    calls = []

    def fake_norm(value, *, gamma, epsilon):
        calls.append(("norm", value, None, gamma, epsilon))
        return output, torch.empty(0)

    def fake_add_norm(value, skip, gamma, epsilon):
        calls.append(("add_norm", value, skip, gamma, epsilon))
        return output, torch.empty(0), updated

    backend = types.ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_rms_norm", fake_norm, raising=False)
    monkeypatch.setattr(backend, "npu_add_rms_norm", fake_add_norm, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    result = norm(x, residual) if with_residual else norm(x)
    if with_residual:
        assert result[0] is output
        assert result[1] is updated
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)
    else:
        assert result is output
    assert len(calls) == 1
    kind, value, skip, gamma, epsilon = calls[0]
    assert kind == ("add_norm" if with_residual else "norm")
    assert value is x and skip is residual and gamma is norm.weight
    assert epsilon == 1e-5
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(norm.weight, weight, rtol=0, atol=0)


@pytest.mark.parametrize("norm_cls", [RMSNorm, QKRMSNorm])
def test_npu_method_falls_back_for_cpu_input(norm_cls) -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 8
    norm = norm_cls(4)
    torch.testing.assert_close(norm.forward_npu(x), norm.forward_native(x), rtol=0, atol=0)


def test_vae_keeps_native_dispatch_on_npu(monkeypatch) -> None:
    platform = custom_op.current_omni_platform
    monkeypatch.setattr(platform, "is_rocm", lambda: False)
    monkeypatch.setattr(platform, "is_cuda", lambda: False)
    monkeypatch.setattr(platform, "is_npu", lambda: True)

    def unexpected_fusion(*args, **kwargs):
        pytest.fail("LiDAR VAE must not enter the shared fused NPU path")

    monkeypatch.setattr(shared_norm.RMSNorm, "forward_npu", unexpected_fusion)
    norm = VAERMSNorm(4)
    assert norm._forward_method.__func__ is VAERMSNorm.forward_npu
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 8
    torch.testing.assert_close(norm(x), norm.forward_native(x), rtol=0, atol=0)
