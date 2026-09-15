# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Z-Image attention on the fused single-stream QK RMSNorm + RoPE op."""

import os

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_HEADS, _HEAD_DIM = 30, 128
_DIM = _HEADS * _HEAD_DIM


@pytest.fixture
def _dist_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29519")
    # vLLM 0.28+: parallel-state init, CustomOp construction and the linear
    # layers' forward all read the current vLLM config.
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        yield
        cleanup_dist_env_and_memory()


def test_packed_table_uses_row_zero_like_rotary_embedding(monkeypatch):
    """``RotaryEmbedding`` applies ``cos[0]``/``sin[0]`` to every batch element;
    the packed table must repeat exactly those rows."""
    from vllm_omni.diffusion.models.z_image.z_image_transformer import _packed_qk_norm_rope_table

    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    cos, sin = torch.randn(3, 20, _HEAD_DIM // 2), torch.randn(3, 20, _HEAD_DIM // 2)
    table = _packed_qk_norm_rope_table(cos, sin, torch.bfloat16)
    assert table.shape == (60, _HEAD_DIM) and table.dtype == torch.bfloat16
    expected = torch.cat((cos[0], sin[0]), dim=-1).to(torch.bfloat16)
    for b in range(3):
        assert torch.equal(table[b * 20 : (b + 1) * 20], expected)
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1000000")
    assert _packed_qk_norm_rope_table(cos, sin, torch.bfloat16) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_z_image_attention_fused_matches_eager(_dist_env):
    from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageAttention, _packed_qk_norm_rope_table

    torch.manual_seed(3)
    with torch.device("cuda"):
        attn = ZImageAttention(dim=_DIM, num_heads=_HEADS, num_kv_heads=_HEADS, eps=1e-6)
    attn = attn.to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_((0.02 * torch.randn(p.shape, device="cuda")).to(p.dtype) if p.dim() > 1 else p)
        attn.norm_q.weight.uniform_(0.5, 1.5)
        attn.norm_k.weight.uniform_(0.5, 1.5)
    batch, seq = 2, 320
    hidden = torch.randn(batch, seq, _DIM, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(batch, seq, _HEAD_DIM // 2, device="cuda")
    cos, sin = torch.cos(angles), torch.sin(angles)
    mask = torch.ones(batch, seq, dtype=torch.bool, device="cuda")
    table = _packed_qk_norm_rope_table(cos, sin, torch.bfloat16)

    with torch.no_grad():
        eager = attn(hidden, mask, cos, sin)
        fused = attn(hidden, mask, cos, sin, qk_norm_rope_table=table)
    torch.testing.assert_close(fused, eager, atol=0.05, rtol=0.05)
