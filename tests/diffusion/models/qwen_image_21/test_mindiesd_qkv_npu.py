# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers.fused_norm_rope import FusedNormRope, prepare_rope_tables

pytestmark = [pytest.mark.core_model, pytest.mark.npu]


@pytest.mark.parametrize("seq_len", [17, 1024, 1040, 4096, 4112, 16411])
def test_fused_qkv_matches_model_rope(seq_len):
    pytest.importorskip("torch_npu")
    pytest.importorskip("mindiesd")
    import mindiesd

    if not callable(getattr(mindiesd, "norm_rope_concat", None)):
        pytest.skip("MindIE-SD norm_rope_concat Python interface is not installed")
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        _apply_qwen_image21_rotary_emb_native,
    )
    from vllm_omni.platforms.npu import is_a5

    if not is_a5():
        pytest.skip("This adapter is enabled only on A5")
    generator = torch.Generator().manual_seed(42)
    packed = torch.randn(1, seq_len, 3 * 4 * 128, generator=generator).to(device="npu", dtype=torch.bfloat16)
    q, k, v = [x.unflatten(-1, (4, 128)) for x in packed.chunk(3, -1)]
    w = torch.linspace(0.5, 1.5, 128, device="npu", dtype=q.dtype)
    phase = torch.randn(seq_len, 64, generator=generator).to("npu")
    freqs = torch.polar(torch.ones_like(phase), phase)
    sin, cos = prepare_rope_tables(freqs, q.dtype)
    actual = FusedNormRope().forward_npu(q, k, v, w, w, 1e-6, (sin, cos))
    # Independent FP32 mathematical reference for the fused contract.
    for name, x, out in zip(("q", "k"), (q, k), actual[:2]):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        normalized = normalized * w.float()
        rotated = torch.stack((-normalized[..., 1::2], normalized[..., 0::2]), dim=-1).flatten(-2)
        expected = (normalized * cos[None, :, None].float() + rotated * sin[None, :, None].float()).to(x.dtype)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        math_delta = out.float() - expected.float()
        math_relative_l2 = math_delta.norm() / expected.float().norm()
        original = _apply_qwen_image21_rotary_emb_native(torch.ops.npu.npu_rms_norm(x, w, epsilon=1e-6)[0], freqs)
        delta = out.float() - original.float()
        relative_l2 = delta.norm() / original.float().norm()
        print(
            f"seq={seq_len} {name} math_max_abs={math_delta.abs().max().item():.6g} "
            f"math_relative_l2={math_relative_l2.item():.6g} "
            f"original_max_abs={delta.abs().max().item():.6g} "
            f"original_relative_l2={relative_l2.item():.6g}"
        )
    torch.testing.assert_close(actual[2], v, atol=0, rtol=0)
