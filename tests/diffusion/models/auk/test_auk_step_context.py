# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AuK DiT's per-request prepare() / per-step step() split.

The Euler loop prepares the conditioning once and then only steps; these
tests pin that the split, the additive key-padding bias and the shared CFG
target embedding reproduce the one-shot forward exactly.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.auk.auk_transformer import (
    AuKTransformer,
    _key_padding_bias,
    _sdpa,
    build_time_grid,
    sample_latents,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_dit(attn_mask_enabled: bool = True) -> AuKTransformer:
    torch.manual_seed(3)
    return AuKTransformer(
        dim=32,
        heads=2,
        dim_head=16,
        ff_mult=2,
        latent_dim=4,
        text_hidden_dim=8,
        num_layers=2,
        num_single_layers=2,
        attn_mask_enabled=attn_mask_enabled,
    ).eval()


def _inputs(ref_frames: int, pad: bool) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator().manual_seed(5)
    text = torch.randn(1, 7, 8, generator=generator)
    c_mask = torch.ones(1, 7, dtype=torch.bool)
    mask = None
    ref = torch.randn(1, ref_frames, 4, generator=generator)
    ref_mask = torch.ones(1, ref_frames, dtype=torch.bool)
    if pad:
        # Padded tails in every stream, as the CUDA graph buckets produce.
        c_mask[:, 5:] = False
        text[:, 5:] = 0.0
        mask = torch.ones(1, 9, dtype=torch.bool)
        mask[:, 7:] = False
        if ref_frames:
            ref_mask[:, ref_frames - 1 :] = False
    return {"text": text, "c_mask": c_mask, "mask": mask, "ref": ref, "ref_mask": ref_mask}


@torch.inference_mode()
def test_key_padding_bias_matches_boolean_mask() -> None:
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 3, 6, 8, generator=generator) for _ in range(3))
    keep = torch.tensor([[True] * 6, [True, True, True, True, False, False]])
    expanded = keep[:, None, None, :].expand(2, 3, 6, 6)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=expanded)
    torch.testing.assert_close(_sdpa(q, k, v, _key_padding_bias(keep, q.dtype)), want)
    assert _key_padding_bias(None, q.dtype) is None


@torch.inference_mode()
@pytest.mark.parametrize("attn_mask_enabled", [True, False])
@pytest.mark.parametrize("cfg_infer", [False, True])
@pytest.mark.parametrize("ref_frames", [0, 4])
@pytest.mark.parametrize("pad", [False, True])
def test_prepare_then_step_matches_forward(attn_mask_enabled: bool, cfg_infer: bool, ref_frames: int, pad: bool):
    dit = _make_dit(attn_mask_enabled)
    inputs = _inputs(ref_frames, pad)
    x = torch.randn(1, 9, 4, generator=torch.Generator().manual_seed(9))
    time = torch.tensor(0.3)

    want = dit(x, time=time, cfg_infer=cfg_infer, **inputs)
    ctx = dit.prepare(inputs.pop("text"), target_len=x.shape[1], cfg_infer=cfg_infer, **inputs)
    got = dit.step(x, time, ctx)

    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got.shape == (2 if cfg_infer else 1, 9, 4)
    assert ctx.branches == (2 if cfg_infer else 1)
    if not attn_mask_enabled:
        assert ctx.joint_bias is None and ctx.single_bias is None


@torch.inference_mode()
def test_cfg_branches_match_separate_cond_and_uncond_forwards() -> None:
    """The shared target embedding keeps each CFG branch equal to its own forward."""
    dit = _make_dit()
    inputs = _inputs(4, pad=True)
    x = torch.randn(1, 9, 4, generator=torch.Generator().manual_seed(9))
    time = torch.tensor(0.6)

    both = dit(x, time=time, cfg_infer=True, **inputs)
    cond = dit(x, time=time, **inputs)
    uncond = dit(x, time=time, drop_audio_cond=True, drop_text=True, **inputs)

    torch.testing.assert_close(both[:1], cond)
    torch.testing.assert_close(both[1:], uncond)


@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_hoisted_euler_loop_matches_per_step_forward(cfg_strength: float) -> None:
    """sample_latents prepares once; stepping the full forward every time agrees."""
    dit = _make_dit()
    inputs = _inputs(4, pad=False)
    inputs.pop("mask")
    grid = [0.0, 0.3, 0.7, 1.0]

    got = sample_latents(
        dit, **inputs, gen_frames=9, t_grid=grid, cfg_strength=cfg_strength, generator=torch.Generator().manual_seed(1)
    )

    x = torch.randn(9, 4, generator=torch.Generator().manual_seed(1)).unsqueeze(0)
    timesteps = build_time_grid(nfe=len(grid) - 1, sway_sampling_coef=None, t_grid=grid, device="cpu")
    for i in range(len(grid) - 1):
        if cfg_strength >= 1e-5:
            cond, uncond = dit(x, time=timesteps[i], cfg_infer=True, **inputs).chunk(2, dim=0)
            v = cond + (cond - uncond) * cfg_strength
        else:
            v = dit(x, time=timesteps[i], **inputs)
        x = x + (timesteps[i + 1] - timesteps[i]) * v

    # sample_latents precomputes the adaLN modulations of the whole grid in one
    # GEMM per layer, whose rows round differently from one-row GEMMs in fp32.
    torch.testing.assert_close(got, x, rtol=0, atol=1e-6)
    assert dit.text_cond is None and dit.text_uncond is None


@torch.inference_mode()
def test_context_copy_rejects_structural_mismatch() -> None:
    dit = _make_dit()
    inputs = _inputs(4, pad=True)
    text = inputs.pop("text")
    cfg = dit.prepare(text, target_len=9, cfg_infer=True, **inputs)
    plain = dit.prepare(text, target_len=9, **inputs)
    with pytest.raises(ValueError, match="branches"):
        cfg.copy_(plain)

    clone = cfg.clone()
    other = dit.prepare(text * 2.0, target_len=9, cfg_infer=True, **inputs)
    clone.copy_(other)
    torch.testing.assert_close(clone.c, other.c)
    assert clone.c.data_ptr() != other.c.data_ptr()
