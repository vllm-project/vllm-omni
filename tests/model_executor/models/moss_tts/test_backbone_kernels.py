# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backbone kernels: RMSNorm with residual, SiLU-and-mul, Q/K norm + RoPE against vLLM references."""

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.tts,
    *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _rms_norm_reference(x, weight, eps, residual=None):
    # vllm.ir.ops.rms_norm / fused_add_rms_norm
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
        residual = xf.to(x.dtype)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = (xf.to(weight.dtype) * weight).to(x.dtype)
    return out if residual is None else (out, residual)


@pytest.mark.parametrize("tokens", [1, 64, 300])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rms_norm_matches_vllm_reference(tokens, with_residual):
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_backbone import rms_norm

    torch.manual_seed(0)
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    weight = torch.rand(2560, device="cuda", dtype=torch.bfloat16) + 0.5
    got = rms_norm(x, weight, 1e-6, residual)
    ref = _rms_norm_reference(x, weight, 1e-6, residual)
    if with_residual:
        torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0)
        got, ref = got[0], ref[0]
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("tokens", [1, 64, 300])
def test_silu_and_mul_matches_reference(tokens):
    import torch.nn.functional as F

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_backbone import silu_and_mul

    torch.manual_seed(0)
    x = torch.randn(tokens, 2 * 9728, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(silu_and_mul(x), F.silu(x[:, :9728]) * x[:, 9728:], atol=1e-3, rtol=1.6e-2)


@pytest.mark.parametrize("tokens", [1, 64, 300])
def test_qk_norm_rope_matches_reference(tokens):
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_backbone import qk_norm_rope_

    torch.manual_seed(0)
    hq, hk, d = 32, 8, 128
    qkv = torch.randn(tokens, (hq + 2 * hk) * d, device="cuda", dtype=torch.bfloat16)
    qw = torch.rand(d, device="cuda", dtype=torch.bfloat16) + 0.5
    kw = torch.rand(d, device="cuda", dtype=torch.bfloat16) + 0.5
    positions = torch.randint(0, 4096, (tokens,), device="cuda")
    inv_freq = 1.0 / (1_000_000 ** (torch.arange(0, d, 2, device="cuda", dtype=torch.float32) / d))
    freqs = torch.arange(4096, device="cuda", dtype=torch.float32)[:, None] * inv_freq[None, :]
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)

    q, k, v = qkv.split([hq * d, hk * d, hk * d], -1)
    qn = _rms_norm_reference(q.reshape(tokens, hq, d), qw, 1e-6)
    kn = _rms_norm_reference(k.reshape(tokens, hk, d), kw, 1e-6)
    cos, sin = cache[positions].float().chunk(2, dim=-1)

    def rope(x):
        x1, x2 = x.float().chunk(2, dim=-1)
        c, s = cos[:, None], sin[:, None]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)

    out = qkv.clone()
    qk_norm_rope_(out, positions, hq, hk, d, 1e-6, qw, kw, cache)
    fq, fk, fv = out.split([hq * d, hk * d, hk * d], -1)
    torch.testing.assert_close(fq.reshape(tokens, hq, d), rope(qn), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fk.reshape(tokens, hk, d), rope(kn), atol=2e-2, rtol=2e-2)
    assert torch.equal(fv, v)
