# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA platform required"),
]


def test_rmsnorm_cuda_residual_matches_native_contract() -> None:
    eps = 1e-6
    device = torch.device(current_omni_platform.device_type)
    torch.manual_seed(0)
    norm = RMSNorm(64, eps=eps, dtype=torch.bfloat16).to(device)
    x = torch.randn(2, 4, 64, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    x_before = x.clone()
    residual_before = residual.clone()

    expected_output, expected_residual = norm.forward_native(x_before, residual_before)
    output, updated_residual = norm.forward_cuda(x, residual)

    torch.testing.assert_close(updated_residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)
    torch.testing.assert_close(residual, residual_before, atol=0, rtol=0)
