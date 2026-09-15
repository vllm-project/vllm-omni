# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Geometry contract RAINFUSION_ATTN enforces before handing a forward to rf_v2.

The grids here are the ones MiniMax-H3 FL2VA actually produces for an 8.7s clip:
1280x768 gives a 62x24x40 latent grid whose 59520 video rows land on the 128-row
kernel block, and 1344x768 gives 62x24x42 whose 62496 rows do not. Both are
handed to MindIE-SD: its rf_v2 preprocessing promotes an irregular video tail
to the always-kept prefix segment before generating the block mask.
"""

import dataclasses
import sys
import types
from unittest import mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends import rainfusion_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout, VideoTokenSpan
from vllm_omni.diffusion.attention.backends.rainfusion_attn import (
    _BLOCK_SIZE,
    RainFusionAttentionBackend,
    RainFusionAttentionImpl,
    RainFusionPlan,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu]

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


def test_misaligned_video_segment_runs_sparse_with_updated_mindiesd():
    plan = make_impl()._resolve_plan(make_metadata(MISALIGNED_GRID))

    assert plan is not None
    assert plan.prefix_len == PREFIX_ROWS
    assert plan.used_len == PREFIX_ROWS + 62496
    assert plan.latent_shape == list(MISALIGNED_GRID)


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


def test_ref2va_multi_video_spans_resolve():
    layout = VideoTokenLayout(
        used_len=12000,
        video_spans=(
            VideoTokenSpan(start=128, latent_grid=(4, 16, 64), role="reference"),
            VideoTokenSpan(start=5000, latent_grid=(4, 16, 64), role="target"),
        ),
    )
    plan = make_impl()._resolve_plan(AttentionMetadata(extra={"max_seqlen_q": 12000}, video_layout=layout))

    assert plan is not None
    assert plan.used_len == 12000
    assert plan.video_spans == [
        {"start": 128, "latent_shape": [4, 16, 64]},
        {"start": 5000, "latent_shape": [4, 16, 64]},
    ]


def test_invalid_ref2va_video_spans_fall_back_to_dense():
    layout = VideoTokenLayout(
        used_len=12000,
        video_spans=(
            VideoTokenSpan(start=128, latent_grid=(4, 16, 64), role="reference"),
            VideoTokenSpan(start=4000, latent_grid=(4, 16, 64), role="target"),
        ),
    )
    assert make_impl()._resolve_plan(AttentionMetadata(extra={"max_seqlen_q": 12000}, video_layout=layout)) is None


def test_validate_available_rejects_legacy_mindiesd(monkeypatch):
    mindiesd = types.ModuleType("mindiesd")

    def sparse_attention(query, key, value, **kwargs):
        return query

    mindiesd.sparse_attention = sparse_attention
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)
    monkeypatch.setattr("importlib.util.find_spec", lambda _: object())

    with pytest.raises(ValueError, match="video_spans"):
        RainFusionAttentionBackend.validate_available()


def test_validate_available_accepts_new_mindiesd(monkeypatch):
    mindiesd = types.ModuleType("mindiesd")

    def sparse_attention(query, key, value, *, video_spans=None, **kwargs):
        return query

    mindiesd.sparse_attention = sparse_attention
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)
    monkeypatch.setattr("importlib.util.find_spec", lambda _: object())

    RainFusionAttentionBackend.validate_available()


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


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="rf_v2 runs on Ascend NPU only.")
def test_irregular_video_tail_reproduces_dense_attention_with_updated_mindiesd():
    """vLLM forwards the full irregular grid; MindIE-SD owns its tail handling."""
    torch.manual_seed(1)
    prefix_len, grid = 384, (4, 16, 10)
    video_len = grid[0] * grid[1] * grid[2]
    used = prefix_len + video_len
    heads, head_dim = 4, 128
    q, k, v = (torch.randn(1, used, heads, head_dim, dtype=torch.bfloat16, device="npu") for _ in range(3))

    # Explicit bf16: the dense comparison exercises the plain sparse kernel,
    # and older MindIE-SD releases do not accept precision= (gate would raise).
    impl = make_impl(precision="bf16")
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


# end_step tail fallback: the last ``end_step`` denoise steps must stay dense.


def test_end_step_tail_window_stays_dense():
    """step_idx inside the last end_step window must resolve no sparse plan."""
    impl = make_impl(end_step=3)
    fc = mock.Mock()
    fc.denoise_step_idx = 47
    fc.total_denoise_steps = 50
    with (
        mock.patch.object(rainfusion_attn, "is_forward_context_available", return_value=True),
        mock.patch.object(rainfusion_attn, "get_forward_context", return_value=fc),
    ):
        assert impl._resolve_plan(make_metadata(ALIGNED_GRID)) is None


def test_end_step_before_tail_window_still_sparse():
    """step_idx before the tail window must still resolve a sparse plan."""
    impl = make_impl(end_step=3)
    fc = mock.Mock()
    fc.denoise_step_idx = 46
    fc.total_denoise_steps = 50
    with (
        mock.patch.object(rainfusion_attn, "is_forward_context_available", return_value=True),
        mock.patch.object(rainfusion_attn, "get_forward_context", return_value=fc),
    ):
        assert impl._resolve_plan(make_metadata(ALIGNED_GRID)) is not None


def test_end_step_zero_never_triggers_tail_fallback():
    """end_step=0 must not fall back even on the final denoise step."""
    impl = make_impl(end_step=0)
    fc = mock.Mock()
    fc.denoise_step_idx = 49
    fc.total_denoise_steps = 50
    with (
        mock.patch.object(rainfusion_attn, "is_forward_context_available", return_value=True),
        mock.patch.object(rainfusion_attn, "get_forward_context", return_value=fc),
    ):
        assert impl._resolve_plan(make_metadata(ALIGNED_GRID)) is not None


# precision capability gate: a mindiesd without sparse_attention(precision=)
# must fail fast instead of silently running the BF16 path.


def _fake_mindiesd_module():
    """Install a minimal stand-in so the import inside _forward_sparse_npu succeeds."""
    import types

    fake = types.ModuleType("mindiesd")
    fake.sparse_attention = lambda *args, **kwargs: None
    return fake


def test_precision_non_bf16_requires_mindiesd_support():
    """precision != bf16 against a mindiesd lacking the kwarg must raise RuntimeError."""
    import sys

    impl = make_impl(precision="mix")
    sys.modules["mindiesd"] = _fake_mindiesd_module()
    try:
        with mock.patch.object(rainfusion_attn, "_mindiesd_supports_precision", return_value=False):
            with pytest.raises(RuntimeError, match="requires MindIE-SD"):
                impl._forward_sparse_npu(None, None, None, None)
    finally:
        sys.modules.pop("mindiesd", None)


def test_precision_non_bf16_passes_gate_when_supported():
    """precision != bf16 with a capable mindiesd must not raise from the gate."""
    import sys

    impl = make_impl(precision="mix")
    sys.modules["mindiesd"] = _fake_mindiesd_module()
    try:
        with mock.patch.object(rainfusion_attn, "_mindiesd_supports_precision", return_value=True):
            # q/k/v shapes: [B, S, N, D]; plan geometry must match S.
            q = torch.randn(1, 8, 4, 128)
            plan = RainFusionPlan(prefix_len=0, used_len=8, latent_shape=[2, 2, 2])
            # The gate must pass; the fake mindiesd returns None so no crash.
            out = impl._forward_sparse_npu(q, q, q, plan)
            assert out is None
    finally:
        sys.modules.pop("mindiesd", None)
