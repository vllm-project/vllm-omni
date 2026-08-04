# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.layers.rope import RotaryEmbedding as DiffusionRotaryEmbedding
from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3RotaryEmbedding,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _reference_hunyuan_image3_rope(
    rotary_emb: HunyuanImage3RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.dim() == 2 and positions.shape[0] == 3:
        y_pos = positions[1].float()
        x_pos = positions[2].float()
    else:
        y_pos = positions.float()
        x_pos = positions.float()

    num_tokens = y_pos.shape[0]
    dtype = query.dtype
    query_shape = query.shape
    key_shape = key.shape

    inv_freq = rotary_emb.inv_freq.to(device=y_pos.device, dtype=torch.float32)
    y_freqs = y_pos.unsqueeze(-1) * inv_freq[0::2].unsqueeze(0)
    x_freqs = x_pos.unsqueeze(-1) * inv_freq[1::2].unsqueeze(0)
    freqs = torch.stack([y_freqs, x_freqs], dim=-1).reshape(num_tokens, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(1)
    sin = emb.sin().to(dtype).unsqueeze(1)

    query = query.view(num_tokens, -1, rotary_emb.head_dim)
    key = key.view(num_tokens, -1, rotary_emb.head_dim)
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin
    return query.view(query_shape), key.view(key_shape)


def test_hunyuan_image3_rope_uses_diffusion_rotary_op() -> None:
    rotary_emb = HunyuanImage3RotaryEmbedding(head_dim=64)

    assert isinstance(rotary_emb.rotary_op, DiffusionRotaryEmbedding)


@pytest.mark.parametrize("positions", [torch.arange(7), torch.arange(21).reshape(3, 7)])
def test_hunyuan_image3_rope_matches_manual_interleaved_2d_reference(
    positions: torch.Tensor,
) -> None:
    torch.manual_seed(0)
    rotary_emb = HunyuanImage3RotaryEmbedding(head_dim=64, rope_theta=10000.0)
    query = torch.randn(7, 2 * 64, dtype=torch.float32)
    key = torch.randn(7, 1 * 64, dtype=torch.float32)

    actual_q, actual_k = rotary_emb(positions, query, key)
    expected_q, expected_k = _reference_hunyuan_image3_rope(rotary_emb, positions, query, key)

    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
    assert actual_q.dtype == query.dtype
    assert actual_k.dtype == key.dtype
