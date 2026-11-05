# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA parity for PixelUMM's fused split QK norm/MRoPE and packed local attention.

Every QK fusion tier (Triton, then exact torch.compile) must equal the callers' eager math bit for bit.
"""

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.hy_nomas import decoder as d
from vllm_omni.diffusion.models.hy_nomas import decoder_kernels as k

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")


def use_eager(monkeypatch):
    """Route the QK fusion back to the callers' eager math."""
    monkeypatch.setattr(d, "split_norm_rope", lambda *args: None)


def use_compile_tier(monkeypatch):
    monkeypatch.setattr(k, "_triton_device", lambda x: False)


def forbid_compile_tier(monkeypatch):
    def compiled(name):
        raise AssertionError(f"{name} left the Triton tier")

    monkeypatch.setattr(k, "_compiled", compiled)


def norm_module(width, dtype):
    norm = d.PixelUMMRMSNorm(width, 1e-6).to(device="cuda", dtype=dtype)
    norm.weight.copy_(torch.randn_like(norm.weight))
    return norm


def eager_qk(x, cos, sin, norm_t, norm_hw):
    sections = d.PixelUMMArchitecture(head_dim=x.shape[-1]).rope_sections
    return d.prepare_qk(x, cos, sin, norm_t, norm_hw, sections, x.dtype)


@requires_cuda
@pytest.mark.parametrize(
    "tier,batch,heads,dim,dtype,stacked",
    [
        ("triton", 1, 8, 128, torch.bfloat16, False),
        ("triton", 1, 32, 128, torch.bfloat16, False),
        ("triton", 2, 8, 128, torch.bfloat16, False),
        # Q and K as strided views of a stacked QKV projection.
        ("triton", 2, 32, 128, torch.bfloat16, True),
        ("triton", 2, 8, 128, torch.bfloat16, True),
        ("compile", 2, 8, 64, torch.bfloat16, False),
        ("compile", 1, 8, 128, torch.float16, True),
    ],
)
@torch.inference_mode()
def test_decoder_split_qk_norm_rope_exact(tier, batch, heads, dim, dtype, stacked, monkeypatch):
    torch.manual_seed(42)
    grid = (3, 10, 14)
    if stacked:
        # 32 query heads followed by 8 key and 8 value heads, as QKVParallelLinear lays them out.
        qkv = torch.randn(batch, 420, (32 + 16) * dim, device="cuda", dtype=dtype)
        if heads == 32:
            first = 0
        else:
            first = 32 * dim
        x = qkv[..., first : first + heads * dim].view(batch, 420, heads, dim)
        assert not x.is_contiguous()
    else:
        x = torch.randn(batch, 420, heads, dim, device="cuda", dtype=dtype)
    x[:, 0].zero_()
    norm_t = norm_module(dim // 2, dtype)
    norm_hw = norm_module(dim // 2, dtype)
    cos, sin = d.mrope_tables(d.PixelUMMArchitecture(head_dim=dim), grid, "cuda")
    with pytest.MonkeyPatch.context() as tier_patch:
        if tier == "triton":
            forbid_compile_tier(tier_patch)
        else:
            use_compile_tier(tier_patch)
        actual = k.split_norm_rope(x, cos, sin, norm_t.weight, norm_hw.weight, 1e-6)
    assert actual is not None
    use_eager(monkeypatch)
    torch.testing.assert_close(actual, eager_qk(x, cos, sin, norm_t, norm_hw), rtol=0, atol=0)


def _free_gib():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.mem_get_info()[0] / 2**30


@requires_cuda
@pytest.mark.skipif(_free_gib() < 10, reason="needs about 10 GiB of free GPU memory")
@torch.inference_mode()
def test_decoder_triton_offsets_past_int32(monkeypatch):
    """Batched windows exceed 2**31 elements; the tail must match eager rather than wrap around."""
    torch.manual_seed(42)
    forbid_compile_tier(monkeypatch)
    tokens = 2**31 // (32 * 128) + 64
    q = torch.randn(1, tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    cos, sin = d.mrope_tables(d.PixelUMMArchitecture(), (1, 1, tokens), "cuda")
    norm_t = norm_module(64, torch.bfloat16)
    norm_hw = norm_module(64, torch.bfloat16)
    actual = k.split_norm_rope(q, cos, sin, norm_t.weight, norm_hw.weight, 1e-6)[:, -128:]
    use_eager(monkeypatch)
    expected = eager_qk(q[:, -128:], cos[:, -128:], sin[:, -128:], norm_t, norm_hw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
