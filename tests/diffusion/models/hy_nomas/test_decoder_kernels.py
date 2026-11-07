# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA parity for PixelUMM's fused split QK norm/MRoPE and packed local attention.

Every QK fusion tier (Triton, then exact torch.compile) must equal the callers' eager math bit for bit.
"""

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.hy_nomas import decoder as d
from vllm_omni.diffusion.models.hy_nomas import decoder_attention as a
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "grid,tile,batch",
    [
        # 704x1280 decoder window.
        ((10, 44, 80), 8, 1),
        # Partial 2x2 edge tiles, two batch elements.
        ((1, 18, 34), 8, 2),
    ],
)
@torch.inference_mode()
def test_decoder_packed_local_attention_matches_fp32_tiles(grid, tile, batch):
    torch.manual_seed(42)
    cfg = OmniDiffusionConfig(model_class_name="HyNomasPipeline", dtype=torch.bfloat16)
    with set_current_diffusion_config(cfg), set_forward_context(omni_diffusion_config=cfg):
        attention = Attention(
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
            role="hy_nomas.decoder",
            role_category="self",
            skip_sequence_parallel=True,
        )
        _, _, neighborhoods = d.local_attention_layout(grid, tile, "cuda")
        tokens = grid[0] * grid[1] * grid[2]
        query = torch.randn(batch, tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(batch, tokens, 8, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(batch, tokens, 8, 128, device="cuda", dtype=torch.bfloat16)
        plan = a.packed_local_attention(attention, neighborhoods, query[..., 0, 0])
        if plan is None:
            pytest.skip("the selected attention backend has no multi-document packed varlen path")
        actual = plan(attention, query, key, value).float()
    expected = torch.empty_like(actual)
    for (start, end), neighbors in neighborhoods:
        local_key = torch.cat([key[:, first:last] for first, last in neighbors], dim=1).float()
        local_value = torch.cat([value[:, first:last] for first, last in neighbors], dim=1).float()
        expected[:, start:end] = torch.nn.functional.scaled_dot_product_attention(
            query[:, start:end].float().transpose(1, 2),
            local_key.transpose(1, 2),
            local_value.transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2)
    # BF16 output rounding bounds the difference from FP32 attention.
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
