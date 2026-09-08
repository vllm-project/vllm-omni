# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.minimax_h3.encoder import (
    MiniMaxH3Qwen3VLRMSNorm,
    MiniMaxH3Qwen3VLTextDecoderLayer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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


def test_qwen3_vl_decoder_preserves_post_attention_residual_contract() -> None:
    class ConstantAttention(nn.Module):
        def forward(self, hidden_states, position_embeddings):
            del position_embeddings
            return torch.full_like(hidden_states, 2.0)

    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        rms_norm_eps=1e-6,
    )
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    with (
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        layer = MiniMaxH3Qwen3VLTextDecoderLayer(group, config, torch.float32)
    layer.input_layernorm = nn.Identity()
    layer.self_attn = ConstantAttention()
    layer.mlp = nn.Identity()

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    attention_output = torch.full_like(hidden_states, 2.0)
    expected_residual = hidden_states + attention_output
    expected_normalized = RMSNorm.forward_native(layer.post_attention_layernorm, expected_residual)

    with patch.object(
        layer.post_attention_layernorm,
        "forward",
        wraps=layer.post_attention_layernorm.forward_native,
    ):
        output = layer(hidden_states, (torch.empty(0), torch.empty(0)))

    assert isinstance(layer.post_attention_layernorm, MiniMaxH3Qwen3VLRMSNorm)
    torch.testing.assert_close(output, expected_residual + expected_normalized)
