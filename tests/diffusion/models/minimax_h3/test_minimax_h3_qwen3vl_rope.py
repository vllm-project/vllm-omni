# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize("num_q_heads,num_kv_heads,seq_len", [(8, 2, 16), (64, 8, 1)])
def test_qwen3_vl_npu_rope_matches_bnsd_reference(
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
) -> None:
    from vllm_omni.platforms.npu.models.minimax_h3 import _apply_rotary_pos_emb_npu

    device = torch.device("npu")
    dtype = torch.bfloat16
    q = torch.randn(1, num_q_heads, seq_len, 128, device=device, dtype=dtype)
    k = torch.randn(1, num_kv_heads, seq_len, 128, device=device, dtype=dtype)
    cos = torch.randn(1, seq_len, 128, device=device, dtype=dtype)
    sin = torch.randn(1, seq_len, 128, device=device, dtype=dtype)

    q_out, k_out = _apply_rotary_pos_emb_npu(q, k, cos, sin)
    cos_bnsd = cos.unsqueeze(1)
    sin_bnsd = sin.unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    torch.testing.assert_close(
        q_out,
        q * cos_bnsd + rotate_half(q) * sin_bnsd,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        k_out,
        k * cos_bnsd + rotate_half(k) * sin_bnsd,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
def test_qwen3_vl_npu_rope_preserves_batched_mrope_positions() -> None:
    """Keep per-sample Qwen3-VL M-RoPE frequencies on the BNSD NPU path."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLTextRotaryEmbedding,
    )
    from vllm_omni.platforms.npu.models.minimax_h3 import _apply_rotary_pos_emb_npu

    device = torch.device("npu")
    dtype = torch.bfloat16
    batch_size, seq_len, head_dim = 2, 16, 128
    config = SimpleNamespace(
        hidden_size=head_dim,
        num_attention_heads=1,
        rope_parameters={
            "rope_theta": 1_000_000.0,
            "mrope_section": [24, 20, 20],
        },
    )
    rope = MiniMaxH3Qwen3VLTextRotaryEmbedding(config).to(device)
    positions = torch.arange(seq_len, device=device)
    position_ids = torch.stack(
        (
            torch.stack((positions, positions + 7)),
            torch.stack((positions * 2, positions * 2 + 3)),
            torch.stack((positions * 3, positions * 3 + 5)),
        )
    )
    q = torch.randn(batch_size, 8, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, 2, seq_len, head_dim, device=device, dtype=dtype)
    cos, sin = rope(q, position_ids)

    q_out, k_out = _apply_rotary_pos_emb_npu(q, k, cos, sin)
    cos_bnsd = cos.unsqueeze(1)
    sin_bnsd = sin.unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    assert not torch.equal(cos[0], cos[1])
    torch.testing.assert_close(
        q_out,
        q * cos_bnsd + rotate_half(q) * sin_bnsd,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        k_out,
        k * cos_bnsd + rotate_half(k) * sin_bnsd,
        atol=2e-2,
        rtol=2e-2,
    )
