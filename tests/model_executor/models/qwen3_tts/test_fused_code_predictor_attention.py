# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused residual-codebook attention: q/k norm, RoPE, cache writes and causal attention."""

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _norm_rope(x, w, cos, sin, eps):
    """``_head_norm_rope`` in PyTorch: HF RMSNorm then RoPE, in BF16 elementwise steps."""
    x = x.float()
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    n = _bf(w.float() * _bf(x * r))
    half = n.shape[-1] // 2
    n1, n2 = n[..., :half], n[..., half:]
    c1, c2, s1, s2 = cos[..., :half], cos[..., half:], sin[..., :half], sin[..., half:]
    return torch.cat([_bf(_bf(n1 * c1) + _bf(-n2 * s1)), _bf(_bf(n2 * c2) + _bf(n1 * s2))], -1)


@pytest.mark.parametrize("row_rms", [False, True])
@pytest.mark.parametrize("nq,first", [(2, 0), (1, 2), (1, 9), (1, 15)])
@torch.inference_mode()
def test_fused_attention_matches_reference(nq: int, first: int, row_rms: bool):
    from vllm.triton_utils import triton

    from vllm_omni.model_executor.models.qwen3_tts.fused_code_predictor import _cp_attention_kernel

    torch.manual_seed(nq * 100 + first)
    batch, heads, kv_heads, head_dim, groups = 3, 4, 2, 128, 16
    max_pos, eps, scale = groups + 1, 1e-6, head_dim**-0.5
    width = (heads + 2 * kv_heads) * head_dim
    dev = "cuda"
    qkv = torch.randn(batch * nq, width, device=dev, dtype=torch.bfloat16)
    k_cache = torch.randn(batch, kv_heads, max_pos, head_dim, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    q_w = torch.rand(head_dim, device=dev, dtype=torch.bfloat16) + 0.5
    k_w = torch.rand(head_dim, device=dev, dtype=torch.bfloat16) + 0.5
    inv = 1.0 / (1e6 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    angles = torch.cat([angles, angles], -1)
    cos, sin = angles.cos().to(dev, torch.bfloat16), angles.sin().to(dev, torch.bfloat16)
    hidden, row_eps = 256, 1e-6
    # Residual rows the QKV projection read (RMSNorm weight folded into it): the
    # kernel scales each QKV row by its row's RMSNorm factor.
    x = torch.randn(batch * nq, hidden, device=dev, dtype=torch.bfloat16) * 3
    eff = qkv.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + row_eps) if row_rms else qkv

    # reference
    q = eff[:, : heads * head_dim].reshape(batch, nq, heads, head_dim)
    k = eff[:, heads * head_dim : (heads + kv_heads) * head_dim].reshape(batch, nq, kv_heads, head_dim)
    v = eff[:, (heads + kv_heads) * head_dim :].reshape(batch, nq, kv_heads, head_dim)
    pos = torch.arange(first, first + nq, device=dev)
    c, s = cos[pos].float()[None, :, None], sin[pos].float()[None, :, None]
    k_ref, v_ref = k_cache.float().clone(), v_cache.float().clone()
    k_ref[:, :, pos] = _norm_rope(k, k_w, c, s, eps).permute(0, 2, 1, 3)
    v_ref[:, :, pos] = _bf(v.float()).permute(0, 2, 1, 3)
    qn = _norm_rope(q, q_w, c, s, eps)  # [B, nq, H, D]
    keys = k_ref.repeat_interleave(heads // kv_heads, 1)[:, :, : first + nq]
    vals = v_ref.repeat_interleave(heads // kv_heads, 1)[:, :, : first + nq]
    scores = torch.einsum("bqhd,bhkd->bhqk", qn, keys) * scale
    allowed = torch.arange(first + nq, device=dev)[None, :] <= pos[:, None]
    probs = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    expected = torch.einsum("bhqk,bhkd->bqhd", probs, vals).reshape(batch * nq, heads * head_dim)

    out = torch.empty(batch * nq, heads * head_dim, device=dev, dtype=torch.bfloat16)
    _cp_attention_kernel[(batch, heads, nq)](
        qkv, k_cache, v_cache, q_w, k_w, cos, sin, out, first, eps, scale,
        x_ptr=x, x_stride=x.stride(0), row_eps=row_eps,
        NQ=nq, H=heads, KVH=kv_heads, HD=head_dim, MAXP=max_pos, BK=triton.next_power_of_2(groups),
        HID=hidden, ROW_RMS=row_rms, num_warps=1,
    )  # fmt: skip
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    # Only this call's positions are written; earlier keys stay untouched.
    torch.testing.assert_close(k_cache.float(), k_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v_cache.float(), v_ref, rtol=0, atol=0)
    untouched = [p for p in range(max_pos) if p not in pos.tolist()]
    assert torch.equal(k_cache[:, :, untouched].float(), k_ref[:, :, untouched])


@torch.inference_mode()
def test_silu_mul_rms_matches_reference():
    from vllm.triton_utils import triton

    from vllm_omni.model_executor.models.qwen3_tts.fused_code_predictor import _silu_mul_rms_kernel

    torch.manual_seed(0)
    rows, inter, hidden, eps = 5, 3072, 1024, 1e-6
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 4
    gu = torch.randn(rows, 2 * inter, device="cuda", dtype=torch.bfloat16) * 6
    out = torch.empty(rows, inter, device="cuda", dtype=torch.bfloat16)
    _silu_mul_rms_kernel[(rows, triton.cdiv(inter, 1024))](gu, x, x.stride(0), out, eps, N=inter, BN=1024, HID=hidden)
    r = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    g, u = gu.float()[:, :inter] * r, gu.float()[:, inter:] * r
    expected = _bf(torch.nn.functional.silu(g)) * u
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
