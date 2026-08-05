# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shape broadcasting regression tests for RotaryEmbedding.forward_native.

Covers the batch-dimension shape mismatch fix: forward_native now strips the
batch dim from 3D cos/sin before applying RoPE, consistent with forward_cuda,
forward_hip, and apply_rotary_emb_mindiesd paths.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.layers.rope import RotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_inputs(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    cos_batch: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate x, cos, sin tensors for RoPE testing.

    Args:
        batch: Batch dimension of x.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Head dimension (must be even).
        cos_batch: Batch dimension of cos/sin. Defaults to batch if None.
        dtype: Data type for the tensors.
    """
    if cos_batch is None:
        cos_batch = batch

    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype)
    cos = torch.randn(cos_batch, seq_len, head_dim // 2, dtype=dtype)
    sin = torch.randn(cos_batch, seq_len, head_dim // 2, dtype=dtype)
    return x, cos, sin


class TestRotaryEmbeddingNativeShapeRegression:
    """Regression tests for forward_native shape handling (#8e297d81)."""

    @pytest.fixture
    def rope_neox(self) -> RotaryEmbedding:
        """NeoX-style (non-interleaved) RoPE."""
        return RotaryEmbedding(is_neox_style=True)

    @pytest.fixture
    def rope_gptj(self) -> RotaryEmbedding:
        """GPT-J style (interleaved) RoPE."""
        return RotaryEmbedding(is_neox_style=False)

    # ── 2D cos/sin (no batch dim) ──────────────────────────────────────

    def test_2d_cos_sin_unchanged(self, rope_neox: RotaryEmbedding) -> None:
        """2D cos/sin [S, D/2] pass through unchanged (no-op path)."""
        x, cos, sin = _make_inputs(batch=2, seq_len=16, num_heads=8, head_dim=64, cos_batch=2)
        # Manually drop batch dim to create 2D cos/sin
        cos_2d = cos[0]
        sin_2d = sin[0]
        assert cos_2d.dim() == 2
        output = rope_neox.forward_native(x, cos_2d, sin_2d)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    # ── 3D cos/sin — batch-dim stripping ──────────────────────────────

    @pytest.mark.parametrize("cos_batch", [1, 2, 4])
    def test_3d_cos_sin_no_error(self, rope_neox: RotaryEmbedding, cos_batch: int) -> None:
        """3D cos/sin with batch dim should not raise errors."""
        x, cos, sin = _make_inputs(batch=2, seq_len=16, num_heads=8, head_dim=64, cos_batch=cos_batch)
        output = rope_neox.forward_native(x, cos, sin)
        assert output.shape == x.shape

    @pytest.mark.parametrize("interleaved", [False, True])
    def test_3d_cos_identical_to_2d(self, interleaved: bool) -> None:
        """3D cos/sin with identical batch slices == 2D cos/sin result."""
        rope = RotaryEmbedding(is_neox_style=not interleaved)
        B, S, H, D = 2, 16, 8, 64
        x = torch.randn(B, S, H, D)
        cos_2d = torch.randn(S, D // 2)
        sin_2d = torch.randn(S, D // 2)

        # 3D variant: expand the same 2D data across batch
        cos_3d = cos_2d.unsqueeze(0).expand(B, -1, -1)
        sin_3d = sin_2d.unsqueeze(0).expand(B, -1, -1)

        out_2d = rope.forward_native(x, cos_2d, sin_2d)
        out_3d = rope.forward_native(x, cos_3d, sin_3d)

        torch.testing.assert_close(out_3d, out_2d, atol=0, rtol=0)

    # ── Batch size mismatch — the regression scenario ─────────────────

    @pytest.mark.parametrize(
        "x_batch,cos_batch",
        [
            (2, 1),  # cos has fewer batch elements
            (1, 2),  # cos has more batch elements (takes [0])
            (4, 1),  # large x batch, single cos
        ],
    )
    def test_batch_size_mismatch_regression(
        self,
        rope_neox: RotaryEmbedding,
        x_batch: int,
        cos_batch: int,
    ) -> None:
        """forward_native works when cos batch dim differs from x batch dim.

        This is the exact regression scenario — before the fix, a 3D cos/sin
        with a different batch size than x would cause a shape mismatch error
        inside apply_rotary_emb_torch during broadcast.
        """
        x, cos, sin = _make_inputs(batch=x_batch, seq_len=16, num_heads=8, head_dim=64, cos_batch=cos_batch)
        output = rope_neox.forward_native(x, cos, sin)
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    # ── Batch consistency — all elements get same rotation ────────────

    def test_2d_cos_broadcasts_identically_across_batch(self, rope_neox: RotaryEmbedding) -> None:
        """With 2D cos, identical x slices → identical output slices."""
        B, S, H, D = 4, 16, 8, 64
        x_template = torch.randn(1, S, H, D)
        x = x_template.expand(B, -1, -1, -1)  # all batch elements identical
        cos = torch.randn(S, D // 2)
        sin = torch.randn(S, D // 2)

        output = rope_neox.forward_native(x, cos, sin)

        # All output batch slices must be identical
        for i in range(1, B):
            torch.testing.assert_close(output[i], output[0], atol=0, rtol=0)

    def test_3d_identical_cos_broadcasts_identically_across_batch(self, rope_neox: RotaryEmbedding) -> None:
        """With 3D cos (identical slices), identical x → identical output."""
        B, S, H, D = 4, 16, 8, 64
        x_template = torch.randn(1, S, H, D)
        x = x_template.expand(B, -1, -1, -1)
        cos = torch.randn(1, S, D // 2).expand(B, -1, -1)
        sin = torch.randn(1, S, D // 2).expand(B, -1, -1)

        output = rope_neox.forward_native(x, cos, sin)

        for i in range(1, B):
            torch.testing.assert_close(output[i], output[0], atol=0, rtol=0)

    # ── Output shape / dtype invariants ───────────────────────────────

    def test_output_shape_preserved(self, rope_neox: RotaryEmbedding) -> None:
        """Output shape must match input shape for all cos/sin dimensionalities."""
        B, S, H, D = 2, 32, 4, 128
        x = torch.randn(B, S, H, D)

        # 2D cos/sin
        cos_2d = torch.randn(S, D // 2)
        sin_2d = torch.randn(S, D // 2)
        assert rope_neox.forward_native(x, cos_2d, sin_2d).shape == (B, S, H, D)

        # 3D cos/sin, matching batch
        cos_3d = torch.randn(B, S, D // 2)
        sin_3d = torch.randn(B, S, D // 2)
        assert rope_neox.forward_native(x, cos_3d, sin_3d).shape == (B, S, H, D)

        # 3D cos/sin, different batch
        cos_3d_1 = torch.randn(1, S, D // 2)
        sin_3d_1 = torch.randn(1, S, D // 2)
        assert rope_neox.forward_native(x, cos_3d_1, sin_3d_1).shape == (B, S, H, D)

    def test_output_dtype_preserved(self, rope_neox: RotaryEmbedding) -> None:
        """Output dtype must match input dtype."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x, cos, sin = _make_inputs(batch=2, seq_len=16, num_heads=8, head_dim=64, dtype=dtype)
            output = rope_neox.forward_native(x, cos, sin)
            assert output.dtype == dtype, f"dtype mismatch: {output.dtype} != {dtype}"

    def test_output_no_nan_inf(
        self,
        rope_neox: RotaryEmbedding,
        rope_gptj: RotaryEmbedding,
    ) -> None:
        """Output must be finite for both RoPE styles."""
        for rope in (rope_neox, rope_gptj):
            x, cos, sin = _make_inputs(batch=2, seq_len=32, num_heads=4, head_dim=64)
            output = rope.forward_native(x, cos, sin)
            assert not torch.isnan(output).any(), f"NaN in {rope}"
            assert not torch.isinf(output).any(), f"Inf in {rope}"

    def test_output_contiguous(self, rope_neox: RotaryEmbedding) -> None:
        """Output should be contiguous."""
        x, cos, sin = _make_inputs(batch=2, seq_len=16, num_heads=8, head_dim=64)
        output = rope_neox.forward_native(x, cos, sin)
        assert output.is_contiguous()

    # ── CFG-style doubled batch ───────────────────────────────────────

    def test_cfg_style_doubled_batch(self, rope_neox: RotaryEmbedding) -> None:
        """CFG (classifier-free guidance) doubles batch — cos may have
        original batch size while x has 2× batch."""
        B, S, H, D = 2, 16, 8, 64
        # Simulate CFG: x has 2× batch (cond + uncond concatenated)
        x = torch.randn(B * 2, S, H, D)
        # cos/sin only have the original batch
        cos = torch.randn(B, S, D // 2)
        sin = torch.randn(B, S, D // 2)

        output = rope_neox.forward_native(x, cos, sin)
        assert output.shape == (B * 2, S, H, D)

    # ── RotaryEmbeddingWan forward_native is unchanged ────────────────

    def test_wan_forward_native_unchanged(self) -> None:
        """RotaryEmbeddingWan.forward_native still uses its own implementation
        (unflatten + stack + flatten) and should not crash with 3D cos/sin."""
        from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan

        rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
        B, S, H, D = 2, 16, 8, 64
        x = torch.randn(B, S, H, D)
        cos = torch.randn(B, S, 1, D // 2)
        sin = torch.randn(B, S, 1, D // 2)

        output = rope.forward_native(x, cos, sin)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
