# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

import vllm_omni.diffusion.layers.custom_op as custom_op
from vllm_omni.diffusion.models.bagel.mot.mot_layernorm import MoTRMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class _NpuPlatform:
    is_rocm = staticmethod(lambda: False)
    is_cuda = staticmethod(lambda: False)
    is_npu = staticmethod(lambda: True)
    is_xpu = staticmethod(lambda: False)
    is_musa = staticmethod(lambda: False)


@pytest.mark.cpu
@pytest.mark.parametrize("use_mot_routing", [False, True])
def test_mot_rms_norm_npu_falls_back_to_native(monkeypatch, use_mot_routing: bool):
    monkeypatch.setattr(
        custom_op,
        "current_omni_platform",
        _NpuPlatform(),
    )

    layer = MoTRMSNorm(hidden_size=8)
    assert layer._forward_method.__func__ is MoTRMSNorm.forward_npu

    x = torch.randn(6, 8)
    text_indices = torch.tensor([0, 2, 4]) if use_mot_routing else None
    vae_indices = torch.tensor([1, 3, 5]) if use_mot_routing else None

    expected = layer.forward_native(x, text_indices, vae_indices)
    actual = layer(x, text_indices, vae_indices)

    torch.testing.assert_close(actual, expected)


@pytest.mark.npu
def test_mot_rms_norm_npu_fused_cann_matches_native():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")

    torch.manual_seed(42)
    device = torch.device("npu")
    token_count, hidden_size = 32, 3584
    layer = MoTRMSNorm(hidden_size=hidden_size, eps=1e-6).to(device)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(hidden_size, device=device) * 0.25 + 1.0)
        layer.gen_weight.copy_(torch.randn(hidden_size, device=device) * 0.25 + 1.0)

    text_indices = torch.arange(0, token_count, 4, device=device)
    mask = torch.ones(token_count, dtype=torch.bool, device=device)
    mask[text_indices] = False
    vae_indices = torch.arange(token_count, device=device)[mask]

    for input_dtype in (torch.bfloat16, torch.float16):
        x = torch.randn(token_count, hidden_size, device=device, dtype=input_dtype)
        tolerance = 2e-2 if input_dtype == torch.bfloat16 else 2e-3

        with torch.no_grad():
            expected = layer.forward_native(x)
            actual = layer.forward_npu(x)
            torch.npu.synchronize()
        assert actual.dtype == input_dtype
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=tolerance)

        with torch.no_grad():
            expected = layer.forward_native(x, text_indices, vae_indices)
            actual = layer.forward_npu(x, text_indices, vae_indices)
            torch.npu.synchronize()
        assert actual.dtype == input_dtype
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=tolerance)
