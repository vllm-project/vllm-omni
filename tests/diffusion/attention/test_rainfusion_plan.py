# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Geometry contract RAINFUSION_ATTN enforces before handing a forward to rf_v2.

The grids here are the ones MiniMax-H3 FL2VA actually produces for an 8.7s clip:
1280x768 gives a 62x24x40 latent grid whose 59520 video rows land on the 128-row
kernel block, and 1344x768 gives 62x24x42 whose 62496 rows straddle it. Only the
first can run sparse: rf_v2 tiles the sequence as a whole but pools its block
mask from the video and prefix segments separately, so the two only agree on the
block count when the video segment ends on a boundary.
"""

import dataclasses

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
from vllm_omni.diffusion.attention.backends.rainfusion_attn import (
    _BLOCK_SIZE,
    RainFusionAttentionImpl,
    RainFusionPlan,
)
from vllm_omni.platforms import current_omni_platform

PREFIX_ROWS = 710  # 14 text rows + 696 audio rows
ALIGNED_GRID = (62, 24, 40)  # 1280x768 -> 59520 video rows, 465 blocks
MISALIGNED_GRID = (62, 24, 42)  # 1344x768 -> 62496 video rows, 488 blocks + 32 rows


def make_impl(**backend_kwargs):
    return RainFusionAttentionImpl(
        num_heads=8,
        head_size=128,
        softmax_scale=128**-0.5,
        prefix="transformer_blocks.0.attn",
        qkv_layout="BSND",
        backend_kwargs={"sparsity": 0.8, **backend_kwargs},
    )


def make_metadata(grid, prefix_len=PREFIX_ROWS, max_seqlen_q=None):
    video_rows = grid[0] * grid[1] * grid[2]
    return AttentionMetadata(
        extra={"max_seqlen_q": prefix_len + video_rows if max_seqlen_q is None else max_seqlen_q},
        video_layout=VideoTokenLayout(prefix_len=prefix_len, latent_grid=grid),
    )


def test_block_aligned_video_segment_runs_sparse():
    plan = make_impl()._resolve_plan(make_metadata(ALIGNED_GRID))

    assert plan is not None
    assert plan.prefix_len == PREFIX_ROWS
    assert plan.used_len == PREFIX_ROWS + 59520
    assert plan.latent_shape == list(ALIGNED_GRID)


def test_misaligned_video_segment_stays_dense():
    # Realigning the sequence would mean padding it, and rf_v2 ignores attn_mask:
    # pad keys would take a share of every real query's softmax denominator.
    assert make_impl()._resolve_plan(make_metadata(MISALIGNED_GRID)) is None


@pytest.mark.parametrize("prefix_len", [1, 127, _BLOCK_SIZE, 710, 900])
def test_prefix_length_does_not_affect_alignment(prefix_len):
    # Only the video segment has to land on a block boundary; rf_v2 pools the
    # prefix separately and keeps every one of its blocks.
    plan = make_impl()._resolve_plan(make_metadata(ALIGNED_GRID, prefix_len=prefix_len))

    assert plan is not None
    assert plan.prefix_len == prefix_len


def test_sparsity_zero_never_resolves_a_plan():
    assert make_impl(sparsity=0.0)._resolve_plan(make_metadata(ALIGNED_GRID)) is None


def test_missing_video_layout_falls_back_to_dense():
    assert make_impl()._resolve_plan(AttentionMetadata(extra={"max_seqlen_q": 60230})) is None


def test_missing_max_seqlen_falls_back_to_dense():
    metadata = AttentionMetadata(
        video_layout=VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=ALIGNED_GRID),
    )

    assert make_impl()._resolve_plan(metadata) is None


def test_video_segment_must_be_the_tail_of_packed_document_zero():
    metadata = make_metadata(ALIGNED_GRID, max_seqlen_q=PREFIX_ROWS + 59520 + 128)

    assert make_impl()._resolve_plan(metadata) is None


@pytest.mark.parametrize("grid", [(4, 24, 40), (1, 24, 40)])
def test_short_video_stays_dense(grid):
    assert make_impl()._resolve_plan(make_metadata(grid)) is None


@pytest.mark.parametrize("qkv_layout", ["BNSD", "BSH"])
def test_explicitly_wrong_layout_is_rejected(qkv_layout):
    with pytest.raises(ValueError, match="BSND"):
        RainFusionAttentionImpl(
            num_heads=8,
            head_size=128,
            softmax_scale=128**-0.5,
            qkv_layout=qkv_layout,
            backend_kwargs={"sparsity": 0.8},
        )


def test_undeclared_layout_stays_dense():
    # An undeclared layout is not an error -- the layer keeps working, just
    # densely, and the fallback sees the same absent layout plain FLASH_ATTN would.
    impl = RainFusionAttentionImpl(
        num_heads=8,
        head_size=128,
        softmax_scale=128**-0.5,
        prefix="transformer_blocks.0.attn",
        backend_kwargs={"sparsity": 0.8},
    )

    assert impl._resolve_plan(make_metadata(ALIGNED_GRID)) is None
    assert impl.dense_fallback.qkv_layout is None


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="rf_v2 runs on Ascend NPU only.")
def test_fully_populated_mask_reproduces_dense_attention():
    """At sparsity=0 every key block is kept, so rf_v2 must match dense attention.

    This is the guard on the geometry the backend hands the kernel: a wrong
    prefix length or latent grid still produces plausible output under sparse
    selection, but shows up here as a mismatch against SDPA.
    """
    torch.manual_seed(0)
    prefix_len, grid = 384, (4, 16, 64)  # 4096 video rows = 32 blocks
    video_len = grid[0] * grid[1] * grid[2]
    used = prefix_len + video_len
    heads, head_dim = 4, 128
    q, k, v = (torch.randn(1, used, heads, head_dim, dtype=torch.bfloat16, device="npu") for _ in range(3))

    impl = make_impl()
    impl.rainfusion = dataclasses.replace(impl.rainfusion, sparsity=0.0)
    plan = RainFusionPlan(prefix_len=prefix_len, used_len=used, latent_shape=list(grid))
    out = impl._forward_sparse_npu(q, k, v, plan)

    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=impl.softmax_scale,
    ).transpose(1, 2)
    error = (out.float() - reference.float()).abs().mean() / reference.float().abs().mean()
    assert error < 2e-3, f"mean relative error {error:.4%} against dense attention"
