# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-shape check of the MammothModa2 DiT attention on CUDA: the shared
backend (FLASH_ATTN by default, head_dim 120) against the pre-change arithmetic
in bf16, with the padding mask the joint text+image sequence carries."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import TransformerBlock

from .test_dit_attention import _reference_attention

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.cuda,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

# MammothModa2-Preview gen_dit_config: hidden 2520, 21 heads, 7 KV heads, head_dim 120.
DIM, HEADS, KV_HEADS = 2520, 21, 7


@pytest.mark.parametrize("seq", [77 + 4096, 512], ids=["t2i_1024", "short"])
def test_real_shape_matches_previous_arithmetic_bf16(seq):
    torch.manual_seed(0)
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(2, seq, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, seq, dtype=torch.bool, device="cuda")
    mask[0, seq - 300 :] = False
    mask[1, seq - 17 :] = False
    angles = torch.rand(1, seq, block.head_dim, device="cuda") * 6.283
    rotary = (angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16))
    with torch.no_grad():
        got = block.attn(
            hidden_states=hidden, encoder_hidden_states=hidden, attention_mask=mask, image_rotary_emb=rotary
        )
        want = _reference_attention(block.attn, hidden, mask, rotary)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[~mask]) == 0
    diff = (got.float() - want.float()).abs()[mask]
    # bf16 kernels differ in accumulation order; measured on A800: max 1.0e-3, mean 5e-5.
    assert diff.max().item() < 2e-2, diff.max().item()
    assert diff.mean().item() < 1e-3, diff.mean().item()


def test_empty_text_stream_on_the_default_backend():
    """The recipe's text-to-image request with text_guidance_scale > 1 runs the
    context refiner on a zero-token unconditional prompt. Before the guard this
    raised ``RuntimeError: step must be nonzero`` from the FA varlen fallback."""
    torch.manual_seed(0)
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5, modulation=False)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(1, 0, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 0, dtype=torch.bool, device="cuda")
    angles = torch.rand(1, 0, block.head_dim, device="cuda")
    with torch.no_grad():
        out = block.attn(
            hidden_states=hidden,
            encoder_hidden_states=hidden,
            attention_mask=mask,
            image_rotary_emb=(angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)),
        )
    assert out.shape == (1, 0, DIM)
