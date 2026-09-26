# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ovis-Image attention on the fused QK RMSNorm + RoPE ops (double and single blocks)."""

import os

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_HEADS, _HEAD_DIM = 24, 128
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
    os.environ.setdefault("MASTER_PORT", "29518")
    # vLLM 0.28+: parallel-state init, CustomOp construction and the linear
    # layers' forward all read the current vLLM config.
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        yield
        cleanup_dist_env_and_memory()


def _rope(seq_len: int):
    torch.manual_seed(seq_len)
    angles = torch.randn(seq_len, _HEAD_DIM // 2, device="cuda")
    return torch.cos(angles), torch.sin(angles)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("double_stream", [True, False])
def test_ovis_attention_fused_matches_eager(_dist_env, double_stream):
    from vllm_omni.diffusion.models.ovis_image.ovis_image_transformer import (
        QK_NORM_ROPE_TABLE_KEY,
        OvisImageAttention,
        pack_qk_norm_rope_table,
    )

    torch.manual_seed(3)
    with torch.device("cuda"):
        attn = OvisImageAttention(
            query_dim=_DIM,
            heads=_HEADS,
            dim_head=_HEAD_DIM,
            added_kv_proj_dim=_DIM if double_stream else None,
            eps=1e-6,
        )
    attn = attn.to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_((0.02 * torch.randn(p.shape, device="cuda")).to(p.dtype) if p.dim() > 1 else p)
        for n in (attn.norm_q, attn.norm_k) + ((attn.norm_added_q, attn.norm_added_k) if double_stream else ()):
            n.weight.uniform_(0.5, 1.5)
    batch, txt, img = 2, 64, 256
    hidden = torch.randn(batch, img if double_stream else txt + img, _DIM, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(batch, txt, _DIM, device="cuda", dtype=torch.bfloat16) if double_stream else None
    cos, sin = _rope(txt + img)
    table = pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=0)

    with torch.no_grad():
        eager = attn(hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin))
        fused = attn(
            hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin), **{QK_NORM_ROPE_TABLE_KEY: table}
        )
    for e, f in zip(eager if double_stream else (eager,), fused if double_stream else (fused,)):
        torch.testing.assert_close(f, e, atol=0.05, rtol=0.05)
