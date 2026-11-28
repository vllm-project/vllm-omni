# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTensorSym, kFp8DynamicTokenSym

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers.fused_fp8_act_quant import add_rms_norm_fp8_per_token, silu_mul_fp8_per_token

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

_EPS = 1e-6


class _Linear(torch.nn.Module):
    """Stands in for a linear layer; only the advertised input key matters."""

    def __init__(self, key=None):
        super().__init__()
        if key is not None:
            self._input_quant_key = key


class _RMSNorm(torch.nn.Module):
    """FP32 RMSNorm as in vLLM-Omni's native path (Cosmos3)."""

    def __init__(self, hidden, device):
        super().__init__()
        self.weight = torch.nn.Parameter((1 + 0.1 * torch.randn(hidden, device=device)).to(torch.bfloat16))
        self.variance_epsilon = _EPS

    def forward(self, x, residual=None):
        if residual is not None:
            # Like vLLM's RMSNorm: normalize x + residual and return the sum as the new residual.
            x = x + residual
            return self(x), x
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight.float() * h).to(x.dtype)


def _per_token_fp8(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """What the consuming linear does when it quantizes its own input."""
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    return QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)(y.reshape(-1, y.shape[-1]))


@pytest.mark.cpu
@pytest.mark.parametrize("key", [None, kFp8DynamicTensorSym])
def test_unfused_fallback(key):
    g, u, x, r = (torch.randn(3, 5, 64, dtype=torch.bfloat16) for _ in range(4))
    norm = _RMSNorm(64, "cpu")
    out = silu_mul_fp8_per_token(g, u, _Linear(key))
    torch.testing.assert_close(out, F.silu(g) * u, rtol=0, atol=0)
    normed, residual = add_rms_norm_fp8_per_token(x, r, norm, (_Linear(key), _Linear(kFp8DynamicTokenSym)))
    torch.testing.assert_close(residual, r + x, rtol=0, atol=0)
    torch.testing.assert_close(normed, norm(r + x), rtol=0, atol=0)


GPU_DEVICES = [
    pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
    pytest.param("xpu", marks=hardware_marks(res={"xpu": "B60"}, num_cards=1)),
]


def _require_device(name):
    backend = getattr(torch, name, None)
    if backend is None or not backend.is_available():
        pytest.skip(f"requires {name}")
    return torch.device(name)


def _check(qa, y_ref):
    assert isinstance(qa, QuantizedActivation) and qa.quant_key == kFp8DynamicTokenSym
    q_ref, s_ref = _per_token_fp8(y_ref)
    torch.testing.assert_close(qa.scale, s_ref, rtol=1e-6, atol=0)
    q = qa.data.reshape(-1, y_ref.shape[-1])
    if q.numel():
        # The fused RMSNorm reduces in a different order; allow rare one-step FP8 differences.
        assert (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean().item() < 1e-3


@pytest.mark.parametrize("device_name", GPU_DEVICES)
@pytest.mark.parametrize("shape", [(1, 1052, 4096), (3, 12288), (0, 4096)])
def test_silu_mul_fused(device_name, shape):
    device = _require_device(device_name)
    g, u = (torch.randn(shape, device=device).to(torch.bfloat16) for _ in range(2))
    qa = silu_mul_fp8_per_token(g, u, _Linear(kFp8DynamicTokenSym))
    assert qa.data.shape == g.shape
    _check(qa, F.silu(g) * u)


@pytest.mark.parametrize("device_name", GPU_DEVICES)
@pytest.mark.parametrize("shape", [(1, 1052, 4096), (2, 3, 4096), (0, 4096)])
def test_add_rms_norm_fused(device_name, shape):
    device = _require_device(device_name)
    x, r = (torch.randn(shape, device=device).to(torch.bfloat16) for _ in range(2))
    norm = _RMSNorm(shape[-1], device)
    qa, residual = add_rms_norm_fp8_per_token(x, r, norm, (_Linear(kFp8DynamicTokenSym),) * 2)
    torch.testing.assert_close(residual, r + x, rtol=0, atol=0)
    _check(qa, norm(r + x))
