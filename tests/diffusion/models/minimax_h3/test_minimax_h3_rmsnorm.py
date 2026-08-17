# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLRMSNorm


def test_qwen3_vl_rmsnorm_uses_common_fused_rmsnorm_contract() -> None:
    """Qwen keeps BF16 gamma while native fallback accumulates in FP32."""
    eps = 1e-6
    norm = MiniMaxH3Qwen3VLRMSNorm(hidden_size=4, eps=eps, dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=torch.bfloat16))
    x = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]],
        dtype=torch.bfloat16,
    )

    x_fp32 = x.float()
    expected = (x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps) * norm.weight.float()).to(x.dtype)

    assert isinstance(norm, RMSNorm)
    assert norm.weight.dtype == torch.bfloat16
    assert set(norm.state_dict()) == {"weight"}
    torch.testing.assert_close(norm.forward_native(x), expected, atol=0, rtol=0)
