import pytest
import torch

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import (
    ErnieImageEmbedND3,
    _apply_rotary_emb,
    rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestErnieImageRopePositionEmbedding:
    """Test ErnieImage RoPE position embedding functionality"""

    def test_rope_output_shape(self):
        """Verify rope produces correct output shape"""
        dim = 64
        theta = 256
        pos = torch.tensor([1.0, 2.0, 3.0])

        out = rope(pos, dim, theta)

        assert out.shape == (3, dim // 2), f"Expected shape (3, {dim // 2}), got {out.shape}"
        assert out.dtype == torch.float32

    def test_rope_trigonometric_relationship(self):
        """Verify rope output satisfies cos^2 + sin^2 = 1"""
        dim = 64
        theta = 256
        pos = torch.tensor([1.0, 2.0, 3.0])

        out = rope(pos, dim, theta)
        cos_out = out.cos()
        sin_out = out.sin()

        cos_sq_sin_sq = cos_out**2 + sin_out**2
        assert torch.allclose(cos_sq_sin_sq, torch.ones_like(cos_sq_sin_sq), atol=1e-6)

    def test_embed_nd3_output_shape(self):
        """Verify ErnieImageEmbedND3 produces correct output shape"""
        head_dim = 128
        axes_dim = (32, 48, 48)
        theta = 256
        B, S = 2, 100

        pos_embed = ErnieImageEmbedND3(dim=head_dim, theta=theta, axes_dim=axes_dim)
        ids = torch.randn(B, S, 3)

        freqs_cos, freqs_sin = pos_embed(ids)

        expected_dim = sum(axes_dim) // 2
        assert freqs_cos.shape == (B, S, expected_dim), (
            f"Expected shape ({B}, {S}, {expected_dim}), got {freqs_cos.shape}"
        )
        assert freqs_sin.shape == (B, S, expected_dim)

    def test_embed_nd3_different_batch_elements(self):
        """Verify different batch elements produce different embeddings"""
        head_dim = 128
        axes_dim = (32, 48, 48)
        theta = 256
        B, S = 2, 100

        pos_embed = ErnieImageEmbedND3(dim=head_dim, theta=theta, axes_dim=axes_dim)

        ids = torch.zeros(B, S, 3)
        ids[0, :, 0] = 1.0
        ids[1, :, 0] = 2.0

        freqs_cos, freqs_sin = pos_embed(ids)

        assert not torch.allclose(freqs_cos[0], freqs_cos[1]), (
            "Different batch elements should produce different cos embeddings"
        )
        assert not torch.allclose(freqs_sin[0], freqs_sin[1]), (
            "Different batch elements should produce different sin embeddings"
        )

    def test_apply_rotary_emb_output_shape(self):
        """Verify _apply_rotary_emb preserves input shape"""
        B, S, H, D = 2, 100, 32, 128

        x = torch.randn(B, S, H, D)
        freqs_cos = torch.randn(B, S, D // 2)
        freqs_sin = torch.randn(B, S, D // 2)

        out = _apply_rotary_emb(x, freqs_cos, freqs_sin)

        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"

    def test_apply_rotary_emb_no_nan_inf(self):
        """Verify _apply_rotary_emb produces valid values"""
        B, S, H, D = 2, 100, 32, 128

        x = torch.randn(B, S, H, D)
        freqs_cos = torch.randn(B, S, D // 2)
        freqs_sin = torch.randn(B, S, D // 2)

        out = _apply_rotary_emb(x, freqs_cos, freqs_sin)

        assert not torch.isnan(out).any(), "Output should not contain NaN"
        assert not torch.isinf(out).any(), "Output should not contain Inf"

    def test_apply_rotary_emb_preserves_batch_dimension(self):
        """Verify _apply_rotary_emb preserves batch dimension for CFG mode"""
        B, S, H, D = 2, 100, 32, 128

        x = torch.randn(B, S, H, D)

        freqs_cos = torch.zeros(B, S, D // 2)
        freqs_sin = torch.zeros(B, S, D // 2)
        freqs_cos[0] = 1.0
        freqs_cos[1] = 0.5

        out = _apply_rotary_emb(x, freqs_cos, freqs_sin)

        assert not torch.allclose(out[0], out[1]), "Different batch elements should produce different outputs"

    def test_cfg_mode_text_lens_difference(self):
        """Verify CFG mode with different text_lens produces different positional embeddings"""
        head_dim = 128
        axes_dim = (32, 48, 48)
        theta = 256
        B = 2
        N_img = 64
        Tmax = 50

        text_lens = torch.tensor([2, 50])

        text_ids = torch.cat(
            [
                torch.arange(Tmax, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1),
                torch.zeros((B, Tmax, 2)),
            ],
            dim=-1,
        )

        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(8, dtype=torch.float32),
                torch.arange(8, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

        image_ids = torch.cat(
            [
                text_lens.float().view(B, 1, 1).expand(-1, N_img, -1),
                grid_yx.view(1, N_img, 2).expand(B, -1, -1),
            ],
            dim=-1,
        )

        ids = torch.cat([image_ids, text_ids], dim=1)

        pos_embed = ErnieImageEmbedND3(dim=head_dim, theta=theta, axes_dim=axes_dim)
        freqs_cos, freqs_sin = pos_embed(ids)

        assert not torch.allclose(freqs_cos[0], freqs_cos[1]), (
            "CFG mode: uncond and cond should have different positional embeddings"
        )
        assert not torch.allclose(freqs_sin[0], freqs_sin[1]), (
            "CFG mode: uncond and cond should have different positional embeddings"
        )

    def test_apply_rotary_emb_matches_diffusers_style(self):
        """Verify _apply_rotary_emb matches diffusers implementation style"""
        B, S, H, D = 2, 100, 32, 128

        x = torch.randn(B, S, H, D)
        freqs = torch.randn(B, S, D // 2)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()

        out_vllm = _apply_rotary_emb(x, freqs_cos, freqs_sin)

        cos_ = freqs_cos.unsqueeze(2)
        sin_ = freqs_sin.unsqueeze(2)
        cos_ = torch.stack([cos_, cos_], dim=-1).reshape(*cos_.shape[:-1], -1)
        sin_ = torch.stack([sin_, sin_], dim=-1).reshape(*sin_.shape[:-1], -1)
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        out_diffusers = x * cos_ + rotated * sin_

        assert torch.allclose(out_vllm, out_diffusers, atol=1e-6), "vllm-omni and diffusers outputs should match"
