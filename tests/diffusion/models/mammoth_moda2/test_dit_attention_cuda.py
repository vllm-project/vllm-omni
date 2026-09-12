# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-shape check of the MammothModa2 DiT attention on CUDA: the shared
backend (FLASH_ATTN by default, head_dim 120) against the pre-change arithmetic
in bf16, with the padding mask the joint text+image sequence carries."""

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import TransformerBlock

from .test_dit_attention import _reference_attention

pytestmark = [
    pytest.mark.core_model,
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


def test_empty_text_stream_skipped_by_apply_refiners():
    """The recipe's text-to-image request with text_guidance_scale > 1 runs the
    context refiner on a zero-token unconditional prompt. The block-level guard
    was removed so ``TransformerBlock.forward`` stays fullgraph-clean under
    torch.compile; the skip now lives in ``Transformer2DModel._apply_refiners``,
    which runs in eager Python and can safely test ``.shape[1] > 0``. Assert
    that path returns the empty text tensor unchanged instead of running the
    (backend-crashing) block on a zero-length sequence."""
    from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel

    torch.manual_seed(0)
    # Shrunk config: only exercise the context_refiner path; construction cost
    # is kept minimal because we never run the noise/main layers here.
    model = (
        Transformer2DModel(
            patch_size=2,
            in_channels=16,
            hidden_size=192,
            num_layers=1,
            num_refiner_layers=1,
            num_attention_heads=6,
            num_kv_heads=2,
            multiple_of=32,
            ffn_dim_multiplier=1.0,
            norm_eps=1e-5,
            axes_dim_rope=(16, 8, 8),
            axes_lens=(64, 64, 64),
            text_feat_dim=64,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    text_hidden_states = torch.randn(1, 0, model.hidden_size, device="cuda", dtype=torch.bfloat16)
    text_attention_mask = torch.ones(1, 0, dtype=torch.bool, device="cuda")
    context_rotary_emb = (
        torch.zeros(1, 0, sum(model.config.axes_dim_rope), device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, 0, sum(model.config.axes_dim_rope), device="cuda", dtype=torch.bfloat16),
    )
    img_tokens = torch.randn(1, 16, model.hidden_size, device="cuda", dtype=torch.bfloat16)
    img_mask = torch.ones(1, 16, dtype=torch.bool, device="cuda")
    noise_rotary_emb = (
        torch.zeros(1, 16, sum(model.config.axes_dim_rope), device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, 16, sum(model.config.axes_dim_rope), device="cuda", dtype=torch.bfloat16),
    )
    temb = torch.randn(1, min(model.hidden_size, 1024), device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out_text, _ = model._apply_refiners(
            text_hidden_states, text_attention_mask, context_rotary_emb, img_tokens, img_mask, noise_rotary_emb, temb
        )
    # Empty text stream is returned unchanged: the block-level backend crash on
    # zero-length varlen is avoided because ``_apply_refiners`` skips the loop.
    assert out_text.shape == (1, 0, model.hidden_size)
    assert torch.equal(out_text, text_hidden_states)


@pytest.mark.parametrize(
    ("seq", "force_gqa_fallback"),
    [(512, False), (77 + 4096, False), (32, True)],
    ids=["short", "t2i_1024", "unsupported_native_gqa"],
)
def test_fp32_falls_back_to_sdpa_and_matches_reference(monkeypatch, seq, force_gqa_fallback):
    """FP32 keeps its dtype and reaches shared SDPA's GQA compatibility check.

    Cover the long masked shape that otherwise falls back to quadratic math
    attention, and force the unsupported-GQA branch independently of hardware.
    """
    torch.manual_seed(0)
    dt = torch.float32
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5)
        .cuda()
        .to(dt)
        .eval()
    )
    hidden = torch.randn(1, seq, DIM, device="cuda", dtype=dt)
    mask = torch.ones(1, seq, dtype=torch.bool, device="cuda")
    mask[:, -13:] = False
    angles = torch.rand(1, seq, block.head_dim, device="cuda")
    rotary = (angles.cos().to(dt), angles.sin().to(dt))
    capabilities, calls, phases = [], [], []
    can_use_fused_gqa = sdpa_backend.can_sdpa_use_fused_gqa
    sdpa = torch.nn.functional.scaled_dot_product_attention
    strategy = block.attn.omni_attn._get_active_parallel_strategy()
    pre_attention, post_attention = strategy.pre_attention, strategy.post_attention

    def prepare(*args):
        phases.append("pre")
        return pre_attention(*args)

    def restore(*args):
        phases.append("post")
        return post_attention(*args)

    def check_gqa(*args):
        supported = False if force_gqa_fallback else can_use_fused_gqa(*args)
        capabilities.append(supported)
        return supported

    def record_sdpa(query, key, value, **kwargs):
        phases.append("sdpa")
        calls.append((query.shape, key.shape, value.shape, query.dtype, kwargs))
        return sdpa(query, key, value, **kwargs)

    with torch.no_grad(), monkeypatch.context() as patch:
        patch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", check_gqa)
        patch.setattr(torch.nn.functional, "scaled_dot_product_attention", record_sdpa)
        patch.setattr(strategy, "pre_attention", prepare)
        patch.setattr(strategy, "post_attention", restore)
        got = block.attn(hidden, hidden, attention_mask=mask, image_rotary_emb=rotary)

    assert phases == ["pre", "sdpa", "post"], "FP32 must preserve the shared parallel dispatch"
    assert len(capabilities) == len(calls) == 1, "FP32 must use shared SDPA's runtime GQA check"
    q_shape, k_shape, v_shape, dtype, kwargs = calls[0]
    expected_kv_heads = KV_HEADS if capabilities[0] else HEADS
    assert q_shape == (1, HEADS, seq, block.head_dim)
    assert k_shape == v_shape == (1, expected_kv_heads, seq, block.head_dim)
    assert dtype == got.dtype == torch.float32
    assert kwargs["enable_gqa"] == capabilities[0]
    assert kwargs["scale"] == block.attn.scale
    assert kwargs["is_causal"] is False
    assert torch.equal(kwargs["attn_mask"], mask[:, None, None, :])

    with torch.no_grad():
        want = _reference_attention(block.attn, hidden, mask, rotary)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[~mask]) == 0
    diff = (got.float() - want.float()).abs()[mask]
    assert diff.max().item() < 1e-3, diff.max().item()
