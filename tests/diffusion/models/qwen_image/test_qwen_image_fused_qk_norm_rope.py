# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen-Image dual-stream attention on the fused joint QK RMSNorm + RoPE op."""

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_HEAD_DIM = 128
_HEADS = 24
_EPS = 1e-6


def _freqs(seq_len: int, device: str) -> torch.Tensor:
    """Complex64 ``[S, D/2]`` frequencies like ``QwenEmbedRope`` emits."""
    torch.manual_seed(seq_len)
    angles = torch.randn(seq_len, _HEAD_DIM // 2, device=device)
    return torch.polar(torch.ones_like(angles), angles)


def test_packed_table_geometry_and_gate(monkeypatch):
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import _packed_qk_norm_rope_table

    vid, txt = _freqs(30, "cpu"), _freqs(7, "cpu")
    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    table = _packed_qk_norm_rope_table(vid, txt, batch_size=2)
    assert table is not None and table.dtype == torch.float32 and table.shape == (2 * 37, _HEAD_DIM)
    # text rows first, then image rows; [cos | sin] halves; batch-repeated
    joint = torch.cat((txt, vid), dim=0)
    torch.testing.assert_close(table[:37, : _HEAD_DIM // 2], joint.real)
    torch.testing.assert_close(table[:37, _HEAD_DIM // 2 :], joint.imag)
    assert torch.equal(table[:37], table[37:])
    # the gate reads the env var at call time
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1000000")
    assert _packed_qk_norm_rope_table(vid, txt, batch_size=2) is None
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    assert _packed_qk_norm_rope_table(vid, txt, batch_size=1) is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("batch,txt_len,img_len", [(1, 77, 1024), (2, 512, 4096)])
def test_fused_path_matches_qwen_image_eager_chain(batch, txt_len, img_len):
    """Fused op vs the chain the attention runs today: ``nn.RMSNorm`` per
    stream, fp32 complex RoPE (``_apply_qwen_image_rotary_emb``), text-first
    cat. Both round once after an fp32 rotation with fp32 frequencies, so the
    two differ only through reduction order in ``sum(x^2)``."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_joint_qkv_norm_rope
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        _apply_qwen_image_rotary_emb,
        _packed_qk_norm_rope_table,
    )

    torch.manual_seed(5)
    dim = _HEADS * _HEAD_DIM

    def qkv(seq_len):
        proj = torch.randn(batch, seq_len, 3 * dim, device="cuda", dtype=torch.bfloat16)
        q, k, v = proj.split([dim, dim, dim], dim=-1)
        return tuple(t.unflatten(-1, (_HEADS, _HEAD_DIM)) for t in (q, k, v))

    txt_q, txt_k, txt_v = qkv(txt_len)
    img_q, img_k, img_v = qkv(img_len)
    norms = [torch.nn.RMSNorm(_HEAD_DIM, eps=_EPS).cuda().to(torch.bfloat16) for _ in range(4)]
    for n in norms:
        n.weight.data.uniform_(0.5, 1.5)
    norm_added_q, norm_added_k, norm_q, norm_k = norms
    vid_freqs, txt_freqs = _freqs(img_len, "cuda"), _freqs(txt_len, "cuda")

    exp_q = torch.cat(
        [
            _apply_qwen_image_rotary_emb(norm_added_q(txt_q), txt_freqs),
            _apply_qwen_image_rotary_emb(norm_q(img_q), vid_freqs),
        ],
        dim=1,
    )
    exp_k = torch.cat(
        [
            _apply_qwen_image_rotary_emb(norm_added_k(txt_k), txt_freqs),
            _apply_qwen_image_rotary_emb(norm_k(img_k), vid_freqs),
        ],
        dim=1,
    )
    exp_v = torch.cat([txt_v, img_v], dim=1)

    table = _packed_qk_norm_rope_table(vid_freqs, txt_freqs, batch)
    act_q, act_k, act_v = fused_joint_qkv_norm_rope(
        txt_q,
        txt_k,
        txt_v,
        img_q,
        img_k,
        img_v,
        norm_added_q.weight,
        norm_added_k.weight,
        norm_q.weight,
        norm_k.weight,
        table,
        _EPS,
    )
    assert torch.equal(act_v, exp_v)
    for act, exp in ((act_q, exp_q), (act_k, exp_k)):
        torch.testing.assert_close(act, exp, atol=0.0625, rtol=0.02)
        # Same rounding order on both sides: at most a sparse 1-ulp residue.
        frac = (act != exp).float().mean().item()
        assert frac < 0.02, f"{frac:.3%} of elements differ"
