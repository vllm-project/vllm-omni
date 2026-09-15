# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""HunyuanVideo 1.5 attention on the fused joint QK RMSNorm + RoPE op."""

import os
from types import SimpleNamespace

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_HEADS, _HEAD_DIM = 16, 128
_DIM = _HEADS * _HEAD_DIM


@pytest.fixture
def _dist_env():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29520")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def test_packed_table_video_rows_then_identity_text_rows(monkeypatch):
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import _packed_qk_norm_rope_table

    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    cos, sin = torch.randn(50, _HEAD_DIM // 2), torch.randn(50, _HEAD_DIM // 2)
    table = _packed_qk_norm_rope_table((cos, sin), text_seq_len=7, batch_size=2, dtype=torch.bfloat16)
    assert table.shape == (2 * 57, _HEAD_DIM) and table.dtype == torch.bfloat16
    half = _HEAD_DIM // 2
    torch.testing.assert_close(table[:50, :half].float(), cos.to(torch.bfloat16).float())
    torch.testing.assert_close(table[:50, half:].float(), sin.to(torch.bfloat16).float())
    assert torch.equal(table[50:57, :half], torch.ones(7, half, dtype=torch.bfloat16))
    assert torch.equal(table[50:57, half:], torch.zeros(7, half, dtype=torch.bfloat16))
    assert torch.equal(table[:57], table[57:])
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1000000")
    assert _packed_qk_norm_rope_table((cos, sin), 7, 2, torch.bfloat16) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_identity_rotation_rows_are_exact():
    """Text rows must come out as plain RMSNorm: cos 1 / sin 0 is exact in the kernel."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import _eager_qk_norm_rope, _launch_fused_qk_norm_rope

    torch.manual_seed(1)
    q = torch.randn(300, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(300, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(_HEAD_DIM, device="cuda", dtype=torch.bfloat16) + 0.5
    ident = (
        torch.cat((torch.ones(300, _HEAD_DIM // 2), torch.zeros(300, _HEAD_DIM // 2)), dim=-1).cuda().to(torch.bfloat16)
    )
    zero_angle = torch.cat((torch.ones(300, _HEAD_DIM // 2), torch.zeros(300, _HEAD_DIM // 2)), dim=-1).cuda()
    fq, fk = _launch_fused_qk_norm_rope(q, k, w, w, ident, 1e-6, interleaved=True)
    eq, ek = _eager_qk_norm_rope(q, k, w, w, zero_angle, 1e-6, _HEAD_DIM, _HEAD_DIM, True)
    torch.testing.assert_close(fq, eq, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(fk, ek, atol=0.0625, rtol=0.02)
    # and against an un-rotated table-free normalization: identical up to reduction order
    ref = torch.nn.functional.rms_norm(q.float(), (_HEAD_DIM,), w.float(), 1e-6).to(torch.bfloat16)
    assert (fq != ref).float().mean().item() < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_hunyuan_attention_fused_matches_eager(_dist_env):
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
        HunyuanVideo15Attention,
        _packed_qk_norm_rope_table,
    )

    torch.manual_seed(3)
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        attn = HunyuanVideo15Attention(
            query_dim=_DIM, heads=_HEADS, dim_head=_HEAD_DIM, added_kv_proj_dim=_DIM, out_dim=_DIM, bias=True, eps=1e-6
        )
    attn = attn.to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_((0.02 * torch.randn(p.shape, device="cuda")).to(p.dtype) if p.dim() > 1 else p)
        for n in (attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k):
            n.weight.uniform_(0.5, 1.5)
    batch, video, txt = 2, 256, 40
    hidden = torch.randn(batch, video, _DIM, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(batch, txt, _DIM, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(video, _HEAD_DIM // 2, device="cuda")
    rope = (torch.cos(angles), torch.sin(angles))
    table = _packed_qk_norm_rope_table(rope, txt, batch, torch.bfloat16)
    config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1, use_hsdp=False))

    with torch.no_grad(), set_forward_context(omni_diffusion_config=config):
        eager = attn(hidden, encoder_hidden_states=encoder, image_rotary_emb=rope)
        fused = attn(hidden, encoder_hidden_states=encoder, image_rotary_emb=rope, qk_norm_rope_table=table)
    for e, f in zip(eager, fused):
        torch.testing.assert_close(f, e, atol=0.05, rtol=0.05)
