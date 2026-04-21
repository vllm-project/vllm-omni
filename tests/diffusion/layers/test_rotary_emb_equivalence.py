# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Numerical equivalence tests for rotary embedding implementations (#2436).

Verifies that the optimized stack+flatten RoPE produces bit-identical results
to the original strided-slice implementation across various tensor shapes and
dtypes, ensuring the refactor is safe.
"""

from __future__ import annotations

import pytest
import torch


def _apply_rotary_emb_helios_original(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Original Helios RoPE using strided slice assignment (pre-#2436)."""
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)


def _apply_rotary_emb_helios_optimized(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Optimized Helios RoPE using stack+flatten (post-#2436)."""
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    rotated = torch.stack(
        (
            x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2],
            x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2],
        ),
        dim=-1,
    )
    return rotated.flatten(-2, -1).type_as(hidden_states)


def _make_inputs(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random hidden_states and freqs_cis for testing."""
    torch.manual_seed(42)
    hidden_states = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype)
    # freqs_cis: [B, seq, head_dim*2] — cos and sin concatenated along last dim
    freqs_cis = torch.randn(batch, seq_len, head_dim * 2, dtype=dtype)
    return hidden_states, freqs_cis


class TestHeliosRoPEEquivalence:
    """Verify optimized Helios RoPE is numerically identical to original."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_equivalence_across_dtypes(self, dtype: torch.dtype) -> None:
        """Optimized output must be bit-identical to original across dtypes."""
        hidden, freqs = _make_inputs(2, 16, 8, 64, dtype=dtype)
        original = _apply_rotary_emb_helios_original(hidden, freqs)
        optimized = _apply_rotary_emb_helios_optimized(hidden, freqs)
        torch.testing.assert_close(optimized, original, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "batch,seq_len,num_heads,head_dim",
        [
            (1, 8, 1, 32),  # minimal: single batch, single head
            (2, 16, 8, 64),  # typical transformer config
            (1, 8192, 4, 64),  # video-scale patch tokens (720p DiT)
            (4, 32, 16, 128),  # large head_dim
        ],
    )
    def test_equivalence_across_shapes(self, batch: int, seq_len: int, num_heads: int, head_dim: int) -> None:
        """Equivalence must hold across different tensor shapes."""
        hidden, freqs = _make_inputs(batch, seq_len, num_heads, head_dim)
        original = _apply_rotary_emb_helios_original(hidden, freqs)
        optimized = _apply_rotary_emb_helios_optimized(hidden, freqs)
        torch.testing.assert_close(optimized, original, atol=0, rtol=0)

    def test_output_contiguous(self) -> None:
        """Optimized output should be contiguous in memory."""
        hidden, freqs = _make_inputs(2, 16, 8, 64)
        optimized = _apply_rotary_emb_helios_optimized(hidden, freqs)
        assert optimized.is_contiguous()

    def test_output_shape_preserved(self) -> None:
        """Output shape must match input shape."""
        hidden, freqs = _make_inputs(2, 16, 8, 64)
        optimized = _apply_rotary_emb_helios_optimized(hidden, freqs)
        assert optimized.shape == hidden.shape

    def test_output_dtype_preserved(self) -> None:
        """Output dtype must match input dtype."""
        hidden, freqs = _make_inputs(2, 16, 8, 64, dtype=torch.float16)
        optimized = _apply_rotary_emb_helios_optimized(hidden, freqs)
        assert optimized.dtype == hidden.dtype

    def test_odd_head_dim_raises(self) -> None:
        """Odd head_dim should fail at unflatten (not a valid RoPE config)."""
        hidden = torch.randn(1, 4, 2, 63)
        freqs = torch.randn(1, 4, 126)
        with pytest.raises(RuntimeError):
            _apply_rotary_emb_helios_optimized(hidden, freqs)
