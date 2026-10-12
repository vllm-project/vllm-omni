# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers.models.embeddings import apply_rotary_emb
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_fused_qk_rope_fake_and_cpu_eligibility():
    from vllm_omni.diffusion.layers.fused_qk_rope import (
        _fused_qk_rope_fake,
        fused_qk_rope,
        fused_qk_rope_supported,
    )

    q = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    cos = torch.randn(3, 8)
    sin = torch.randn_like(cos)

    fake_q, fake_k = _fused_qk_rope_fake(q, k, cos, sin)
    assert fake_q.shape == q.shape and fake_q.dtype == q.dtype
    assert fake_k.shape == k.shape and fake_k.dtype == k.dtype
    assert not fused_qk_rope_supported(q, k, cos, sin)
    with pytest.raises(ValueError, match="contiguous CUDA BF16"):
        fused_qk_rope(q, k, cos, sin)


def _cuda_inputs(batch: int, sequence: int, heads: int = 24, head_dim: int = 128):
    q = torch.randn(batch, sequence, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    # Deliberately do not repeat adjacent table columns.  Diffusers consumes
    # full-width tables, so even and odd outputs may use different values.
    cos = torch.randn(sequence, head_dim, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    return q, k, cos, sin


def _reference(q, k, cos, sin):
    rotary_emb = (cos, sin)
    return (
        apply_rotary_emb(q, rotary_emb, sequence_dim=1),
        apply_rotary_emb(k, rotary_emb, sequence_dim=1),
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("sequence", [512, 4096, 4608])
def test_fused_qk_rope_is_bit_exact_at_longcat_production_shapes(sequence):
    from vllm_omni.diffusion.layers.fused_qk_rope import _launch_fused_qk_rope, fused_qk_rope

    torch.manual_seed(sequence)
    with torch.inference_mode():
        q, k, cos, sin = _cuda_inputs(1, sequence)
        expected = _reference(q, k, cos, sin)
        for actual in (
            _launch_fused_qk_rope(q, k, cos, sin),
            fused_qk_rope(q, k, cos, sin),
        ):
            assert torch.equal(actual[0], expected[0])
            assert torch.equal(actual[1], expected[1])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_qk_rope_is_bit_exact_for_batch_and_full_width_odd_even_tables():
    from vllm_omni.diffusion.layers.fused_qk_rope import fused_qk_rope

    torch.manual_seed(11)
    with torch.inference_mode():
        q, k, cos, sin = _cuda_inputs(2, 17, heads=3)
        assert not torch.equal(cos[:, ::2], cos[:, 1::2])
        assert not torch.equal(sin[:, ::2], sin[:, 1::2])
        expected = _reference(q, k, cos, sin)
        actual = fused_qk_rope(q, k, cos, sin)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_qk_rope_rejects_noncontiguous_and_non_fp32_tables():
    from vllm_omni.diffusion.layers.fused_qk_rope import fused_qk_rope_supported

    q, k, cos, sin = _cuda_inputs(1, 8, heads=2)
    q_noncontiguous = torch.randn(1, 8, 2, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    k_noncontiguous = torch.randn_like(q_noncontiguous)
    cos_noncontiguous = torch.randn(8, 256, device="cuda")[:, ::2]
    sin_noncontiguous = torch.randn(8, 256, device="cuda")[:, ::2]
    with torch.inference_mode():
        assert fused_qk_rope_supported(q, k, cos, sin)
        assert not fused_qk_rope_supported(q_noncontiguous, k_noncontiguous, cos, sin)
        assert not fused_qk_rope_supported(q, k, cos_noncontiguous, sin_noncontiguous)
        assert not fused_qk_rope_supported(q, k, cos.to(torch.bfloat16), sin.to(torch.bfloat16))

        q_wide = torch.randn(1, 8, 2, 256, device="cuda", dtype=torch.bfloat16)
        k_wide = torch.randn_like(q_wide)
        cos_wide = torch.randn(8, 256, device="cuda")
        sin_wide = torch.randn_like(cos_wide)
        assert not fused_qk_rope_supported(q_wide, k_wide, cos_wide, sin_wide)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_qk_rope_custom_op_has_fullgraph_fake():
    from vllm_omni.diffusion.layers.fused_qk_rope import fused_qk_rope

    torch.manual_seed(29)
    with torch.inference_mode():
        q, k, cos, sin = _cuda_inputs(1, 19, heads=3)
        expected = _reference(q, k, cos, sin)
        compiled = torch.compile(fused_qk_rope, dynamic=True, fullgraph=True)
        actual = compiled(q, k, cos, sin)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
