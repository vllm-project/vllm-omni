# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for mindie rope fusion operators.

Tests the RotaryEmbeddingWan integration with mindiesd on NPU platform,
specifically for wan2.2 model usage.
"""

import pytest
import torch

from vllm_omni.diffusion.layers.rope import (
    RotaryEmbedding,
    RotaryEmbeddingWan,
    apply_rotary_emb_mindiesd,
    apply_rotary_emb_torch,
    rotate_half,
)

# Check if NPU is available
is_npu = False
try:
    import torch
    if hasattr(torch, "npu") and torch.npu.is_available():
        is_npu = True
except Exception:
    pass


def create_wan2_2_rope_embeddings(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create rope embeddings following wan2.2's WanRotaryPosEmbed logic.

    This simulates the actual rope embeddings generated in wan2.2 model:
    - rope returns (1, seq_len, 1, head_dim)
    - then sliced as freqs_cos[..., 0::2] and freqs_sin[..., 1::2]
    - resulting in (1, seq_len, 1, head_dim/2)
    """
    seq_len = num_frames * height * width

    # Generate base rope embeddings (similar to WanRotaryPosEmbed)
    # Split dimensions for temporal, height, width
    h_dim = w_dim = 2 * (head_dim // 6)
    t_dim = head_dim - h_dim - w_dim

    freqs_cos_list = []
    freqs_sin_list = []

    for dim in [t_dim, h_dim, w_dim]:
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cos = freqs.cos().float().repeat_interleave(2, dim=-1)
        freqs_sin = freqs.sin().float().repeat_interleave(2, dim=-1)
        freqs_cos_list.append(freqs_cos)
        freqs_sin_list.append(freqs_sin)

    freqs_cos = torch.cat(freqs_cos_list, dim=1)
    freqs_sin = torch.cat(freqs_sin_list, dim=1)

    # Slice as wan2.2 does: freqs_cos[..., 0::2] and freqs_sin[..., 1::2]
    freqs_cos = freqs_cos[..., 0::2]
    freqs_sin = freqs_sin[..., 1::2]

    # Reshape to (1, seq_len, 1, head_dim/2) - the actual shape passed to rope
    freqs_cos = freqs_cos.view(1, seq_len, 1, -1).to(device=device, dtype=dtype)
    freqs_sin = freqs_sin.view(1, seq_len, 1, -1).to(device=device, dtype=dtype)

    return freqs_cos, freqs_sin


def compute_rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """
    Compute RoPE using the reference PyTorch implementation.
    """
    return apply_rotary_emb_torch(x, cos, sin, interleaved=interleaved)


class TestRotateHalf:
    """Test rotate_half function."""

    def test_rotate_half_standard(self):
        """Test standard (non-interleaved) rotation."""
        x = torch.randn(2, 4, 8)
        result = rotate_half(x, interleaved=False)
        assert result.shape == x.shape

    def test_rotate_half_interleaved(self):
        """Test interleaved rotation."""
        x = torch.randn(2, 4, 8)
        result = rotate_half(x, interleaved=True)
        assert result.shape == x.shape


class TestApplyRotaryEmbTorch:
    """Test apply_rotary_emb_torch function."""

    def test_apply_rotary_emb_standard(self):
        """Test standard RoPE application."""
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = apply_rotary_emb_torch(x, cos, sin, interleaved=False)
        assert result.shape == x.shape

    def test_apply_rotary_emb_interleaved(self):
        """Test interleaved RoPE application."""
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = apply_rotary_emb_torch(x, cos, sin, interleaved=True)
        assert result.shape == x.shape


@pytest.mark.skipif(not is_npu, reason="NPU is not available")
class TestApplyRotaryEmbMindiesd:
    """Test apply_rotary_emb_mindiesd function.

    Note: These tests directly call apply_rotary_emb_mindiesd with tensors already on NPU.
    The RotaryEmbedding/RotaryEmbeddingWan classes handle the tensor placement internally.
    """

    def test_mindiesd_rotated_half_2d_cos_sin(self):
        """Test rotated_half mode with 2D cos/sin (S, D/2) - the typical input shape."""
        device = torch.device("npu")
        batch_size, seq_len, num_heads, head_dim = 1, 16, 4, 64

        # Create tensors on NPU
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)

        result = apply_rotary_emb_mindiesd(x, cos, sin, interleaved=False, half_head_dim=True)

        # Compare with reference implementation (also on NPU)
        reference = compute_rope_reference(x, cos, sin, interleaved=False)
        max_diff = torch.max(torch.abs(result - reference)).item()
        mean_diff = torch.mean(torch.abs(result - reference)).item()

        print(f"\nrotated_half 2D cos/sin - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        assert max_diff < 0.01, f"Max diff {max_diff} exceeds threshold"

    def test_mindiesd_rotated_interleaved_2d_cos_sin(self):
        """Test rotated_interleaved mode with 2D cos/sin (S, D/2)."""
        device = torch.device("npu")
        batch_size, seq_len, num_heads, head_dim = 1, 16, 4, 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)

        result = apply_rotary_emb_mindiesd(x, cos, sin, interleaved=True, half_head_dim=True)

        reference = compute_rope_reference(x, cos, sin, interleaved=True)
        max_diff = torch.max(torch.abs(result - reference)).item()
        mean_diff = torch.mean(torch.abs(result - reference)).item()

        print(f"\nrotated_interleaved 2D cos/sin - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        assert max_diff < 0.01, f"Max diff {max_diff} exceeds threshold"

    def test_mindiesd_half_head_dim_false(self):
        """Test with half_head_dim=False (cos/sin size is (S, D)).

        Note: This tests the case where cos/sin has full dimension (S, D).
        """
        device = torch.device("npu")
        batch_size, seq_len, num_heads, head_dim = 1, 16, 4, 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)

        # Just verify it runs without error - mindiesd behavior may differ from torch reference
        result = apply_rotary_emb_mindiesd(x, cos, sin, interleaved=False, half_head_dim=False)
        assert result.shape == x.shape


@pytest.mark.skipif(not is_npu, reason="NPU is not available")
class TestRotaryEmbedding:
    """Test RotaryEmbedding class with NPU backend."""

    def test_rotary_embedding_forward_npu(self):
        """Test RotaryEmbedding forward on NPU with 2D cos/sin."""
        device = torch.device("npu")
        rope = RotaryEmbedding(is_neox_style=False)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

        # Put tensors on NPU
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)

        result = rope.forward_npu(x, cos, sin)
        assert result.shape == x.shape

    def test_rotary_embedding_neox_style(self):
        """Test RotaryEmbedding with neox style on NPU."""
        device = torch.device("npu")
        rope = RotaryEmbedding(is_neox_style=True)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)

        # Neox style uses interleaved=False internally
        result_npu = rope.forward_npu(x, cos, sin)
        result_native = rope.forward_native(x, cos, sin)

        max_diff = torch.max(torch.abs(result_npu - result_native)).item()
        print(f"\nNeox style - Max diff: {max_diff:.6f}")
        assert max_diff < 0.01, f"Max diff {max_diff} exceeds threshold"


@pytest.mark.skipif(not is_npu, reason="NPU is not available")
class TestRotaryEmbeddingWan:
    """Test RotaryEmbeddingWan class with NPU backend.

    Note: The actual usage in wan2.2 is:
    - cos/sin shape: (1, seq_len, 1, head_dim/2) - 4D tensor
    - is_neox_style=False, half_head_dim=True

    RotaryEmbeddingWan.forward_npu handles the 4D->2D reshape internally.
    """

    def test_rotary_embedding_wan_forward_npu(self):
        """Test RotaryEmbeddingWan forward on NPU with half_head_dim=False."""
        device = torch.device("npu")
        rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=False)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)

        result = rope.forward_npu(x, cos, sin)
        assert result.shape == x.shape

    def test_rotary_embedding_wan_half_head_dim(self):
        """Test RotaryEmbeddingWan with half_head_dim=True on NPU.

        Note: forward_native does NOT support half_head_dim=True,
        so we only test that forward_npu runs without error.
        """
        device = torch.device("npu")
        rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)

        # Only test that forward_npu works, don't compare with native
        result_npu = rope.forward_npu(x, cos, sin)
        assert result_npu.shape == x.shape

    def test_rotary_embedding_wan_interleaved(self):
        """Test RotaryEmbeddingWan with interleaved=True on NPU.

        Note: forward_native does NOT support half_head_dim=True,
        so we only test that forward_npu runs without error.
        """
        device = torch.device("npu")
        rope = RotaryEmbeddingWan(is_neox_style=True, half_head_dim=True)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)
        cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)
        sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device=device)

        result_npu = rope.forward_npu(x, cos, sin)
        assert result_npu.shape == x.shape

    def test_rotary_embedding_wan_wan22_style(self):
        """Test RotaryEmbeddingWan with wan2.2 style 4D cos/sin.

        This simulates the actual usage in wan2.2 model:
        - cos/sin shape: (1, seq_len, 1, head_dim/2) - 4D tensor
        - is_neox_style=False, half_head_dim=True

        RotaryEmbeddingWan.forward_npu will reshape 4D to 2D before calling mindiesd.
        """
        device = torch.device("npu")
        batch_size, num_frames, height, width = 1, 2, 4, 4
        seq_len = num_frames * height * width  # 32
        num_heads = 4
        head_dim = 64

        # Create wan2.2 style rope embeddings (4D: 1, S, 1, D/2)
        cos, sin = create_wan2_2_rope_embeddings(
            batch_size, num_frames, height, width, head_dim, device
        )

        # x shape: (batch_size, seq_len, num_heads, head_dim)
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)

        # Create rope with same params as wan2.2
        rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)

        # Apply using NPU (mindiesd) - forward_npu handles 4D->2D reshape
        result_npu = rope.forward_npu(x, cos, sin)
        assert result_npu.shape == x.shape

    def test_rotary_embedding_wan_wan22_i2v_style(self):
        """Test RotaryEmbeddingWan with wan2.2 I2V style (larger resolution).

        This tests a more realistic I2V scenario with larger resolution.
        """
        device = torch.device("npu")
        batch_size, num_frames, height, width = 1, 1, 16, 16
        seq_len = num_frames * height * width  # 256
        num_heads = 8
        head_dim = 128

        # Create wan2.2 style rope embeddings (4D: 1, S, 1, D/2)
        cos, sin = create_wan2_2_rope_embeddings(
            batch_size, num_frames, height, width, head_dim, device
        )

        # x shape: (batch_size, seq_len, num_heads, head_dim)
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=device)

        # Create rope with same params as wan2.2
        rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)

        # Apply using NPU (mindiesd)
        result_npu = rope.forward_npu(x, cos, sin)
        assert result_npu.shape == x.shape


class TestRotaryEmbeddingNative:
    """Test RotaryEmbedding with native (CPU) backend."""

    def test_rotary_embedding_forward_native(self):
        """Test RotaryEmbedding forward_native."""
        rope = RotaryEmbedding(is_neox_style=False)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = rope.forward_native(x, cos, sin)
        assert result.shape == x.shape

    def test_rotary_embedding_forward_native_neox(self):
        """Test RotaryEmbedding forward_native with neox style."""
        rope = RotaryEmbedding(is_neox_style=True)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = rope.forward_native(x, cos, sin)
        assert result.shape == x.shape


if __name__ == "__main__":
    print("Running RoPE Tests...")
    print("=" * 60)
    print(f"NPU available: {is_npu}")
    print("=" * 60)

    # Run all tests
    pytest.main([__file__, "-v", "-s"])