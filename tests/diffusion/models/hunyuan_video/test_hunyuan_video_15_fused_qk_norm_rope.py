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


def _packed_qk_norm_rope_table(rotary_emb, text_seq_len, batch_size, dtype):
    """The model's per-forward table: video rows rotated, text rows identity."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import pack_qk_norm_rope_table
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import _FUSED_MIN_TOKENS

    cos, sin = rotary_emb
    return pack_qk_norm_rope_table(
        cos, sin, batch_size, dtype=dtype, min_tokens=_FUSED_MIN_TOKENS, identity_rows=text_seq_len
    )


@pytest.fixture
def _dist_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29520")
    # vLLM 0.28+: parallel-state init, CustomOp construction and the linear
    # layers' forward all read the current vLLM config.
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        yield
        cleanup_dist_env_and_memory()


def test_packed_table_skipped_on_cpu(monkeypatch):
    """No table (no allocation) where the fused kernel cannot run."""

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    cos, sin = torch.randn(50, _HEAD_DIM // 2), torch.randn(50, _HEAD_DIM // 2)
    assert _packed_qk_norm_rope_table((cos, sin), 7, 2, torch.bfloat16) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_packed_table_video_rows_then_identity_text_rows(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    cos, sin = torch.randn(50, _HEAD_DIM // 2, device="cuda"), torch.randn(50, _HEAD_DIM // 2, device="cuda")
    table = _packed_qk_norm_rope_table((cos, sin), text_seq_len=7, batch_size=2, dtype=torch.bfloat16)
    assert table.shape == (2 * 57, _HEAD_DIM) and table.dtype == torch.bfloat16
    half = _HEAD_DIM // 2
    torch.testing.assert_close(table[:50, :half].float(), cos.to(torch.bfloat16).float())
    torch.testing.assert_close(table[:50, half:].float(), sin.to(torch.bfloat16).float())
    assert torch.equal(table[50:57, :half], torch.ones(7, half, dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(table[50:57, half:], torch.zeros(7, half, dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(table[:57], table[57:])
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1000000")
    assert _packed_qk_norm_rope_table((cos, sin), 7, 2, torch.bfloat16) is None
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    assert _packed_qk_norm_rope_table((cos, sin), 7, 2, torch.float16) is None  # non-bf16 activations


def _identity_table(tokens: int, rotary_half: int) -> torch.Tensor:
    return torch.cat((torch.ones(tokens, rotary_half), torch.zeros(tokens, rotary_half)), dim=-1).to(
        "cuda", torch.bfloat16
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_identity_rotation_rows_are_exact(monkeypatch):
    """Identity rows (cos 1 / sin 0) are bitwise the kernel's un-rotated RMSNorm: a
    rotary_dim = 2 identity table sends 126 of 128 dims through the pass-through branch."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        _launch_fused_joint_qkv_norm_rope,
        _launch_fused_qk_norm_rope,
    )

    torch.manual_seed(1)
    tokens = 300
    q = torch.randn(tokens, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(tokens, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_w = torch.rand(_HEAD_DIM, device="cuda", dtype=torch.bfloat16) + 0.5
    k_w = torch.rand(_HEAD_DIM, device="cuda", dtype=torch.bfloat16) + 0.5
    full_q, full_k = _launch_fused_qk_norm_rope(q, k, q_w, k_w, _identity_table(tokens, _HEAD_DIM // 2), 1e-6, True)
    plain_q, plain_k = _launch_fused_qk_norm_rope(q, k, q_w, k_w, _identity_table(tokens, 1), 1e-6, True)
    assert torch.equal(full_q, plain_q)
    assert torch.equal(full_k, plain_k)

    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    batch, video, txt = 2, 96, 40
    streams = [
        torch.randn(batch, seq, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        for seq in (video, video, video, txt, txt, txt)
    ]
    weights = [torch.rand(_HEAD_DIM, device="cuda", dtype=torch.bfloat16) + 0.5 for _ in range(4)]
    angles = torch.randn(video, _HEAD_DIM // 2, device="cuda")
    table = _packed_qk_norm_rope_table((torch.cos(angles), torch.sin(angles)), txt, batch, torch.bfloat16)
    joint_q, joint_k, joint_v = _launch_fused_joint_qkv_norm_rope(*streams, *weights, table, 1e-6, True)
    text_q, text_k = _launch_fused_qk_norm_rope(
        streams[3].reshape(-1, _HEADS, _HEAD_DIM),
        streams[4].reshape(-1, _HEADS, _HEAD_DIM),
        weights[2],
        weights[3],
        _identity_table(batch * txt, 1),
        1e-6,
        True,
    )
    assert torch.equal(joint_q[:, video:].reshape(-1, _HEADS, _HEAD_DIM), text_q)
    assert torch.equal(joint_k[:, video:].reshape(-1, _HEADS, _HEAD_DIM), text_k)
    assert torch.equal(joint_v[:, video:], streams[5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_hunyuan_attention_fused_matches_eager(_dist_env):
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
        HunyuanVideo15Attention,
    )

    torch.manual_seed(3)
    with torch.device("cuda"):
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
