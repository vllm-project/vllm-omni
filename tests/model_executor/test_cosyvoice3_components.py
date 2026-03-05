# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CosyVoice3 components."""

import pytest
import torch
import torch.nn as nn


class TestPreLookaheadLayer:
    """Tests for PreLookaheadLayer."""

    @pytest.fixture
    def layer(self):
        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer

        return PreLookaheadLayer(in_channels=512, channels=512, pre_lookahead_len=3)

    def test_forward_shape(self, layer):
        """Test that output shape matches input shape."""
        batch, seq_len, channels = 2, 10, 512
        x = torch.randn(batch, seq_len, channels)

        out = layer(x)

        assert out.shape == x.shape

    def test_forward_with_context(self, layer):
        """Test forward with context for streaming."""
        batch, seq_len, channels = 1, 10, 512
        x = torch.randn(batch, seq_len, channels)
        context = torch.randn(batch, 3, channels)  # pre_lookahead_len=3

        layer.eval()
        out = layer(x, context=context)

        assert out.shape == x.shape

    def test_residual_connection(self, layer):
        """Test that residual connection is applied."""
        batch, seq_len, channels = 1, 5, 512
        x = torch.zeros(batch, seq_len, channels)

        # With zero input, output should also be close to zero due to residual
        out = layer(x)

        # Output should be close to input (residual) plus conv output
        assert out.shape == x.shape


class TestDiTAttention:
    """Tests for DiTAttention with diffusion backend."""

    @pytest.fixture
    def attention(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

        return DiTAttention(dim=512, heads=8, dim_head=64, dropout=0.0)

    def test_forward_shape(self, attention):
        """Test attention output shape."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)

        out = attention(x)

        assert out.shape == x.shape

    def test_forward_with_mask(self, attention):
        """Test attention with mask."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        mask[:, -3:] = False  # Mask last 3 positions

        out = attention(x, mask=mask)

        assert out.shape == x.shape
        # Masked positions should be zero
        assert torch.allclose(out[:, -3:], torch.zeros_like(out[:, -3:]))

    def test_qkv_projections(self, attention):
        """Test that Q/K/V projections exist and have correct dimensions."""
        assert hasattr(attention, "to_q")
        assert hasattr(attention, "to_k")
        assert hasattr(attention, "to_v")
        assert attention.to_q.out_features == 512  # heads * dim_head
        assert attention.to_k.out_features == 512
        assert attention.to_v.out_features == 512


class TestDiTBlock:
    """Tests for DiTBlock."""

    @pytest.fixture
    def block(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTBlock

        return DiTBlock(dim=512, heads=8, dim_head=64, ff_mult=4, dropout=0.0)

    def test_forward_shape(self, block):
        """Test block output shape."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)
        t = torch.randn(batch, dim)  # Timestep embedding

        out = block(x, t)

        assert out.shape == x.shape

    def test_adalayernorm_modulation(self, block):
        """Test that AdaLayerNorm modulates based on timestep."""
        batch, seq_len, dim = 1, 8, 512
        x = torch.randn(batch, seq_len, dim)
        t1 = torch.zeros(batch, dim)
        t2 = torch.ones(batch, dim)

        out1 = block(x, t1)
        out2 = block(x, t2)

        # Different timesteps should produce different outputs
        assert not torch.allclose(out1, out2)


class TestDiT:
    """Tests for the full DiT model."""

    @pytest.fixture
    def dit(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT

        return DiT(
            dim=256,
            depth=2,
            heads=4,
            dim_head=64,
            dropout=0.0,
            ff_mult=2,
            mel_dim=80,
            mu_dim=80,
            spk_dim=80,
            long_skip_connection=True,
        )

    def test_forward_shape(self, dit):
        """Test DiT forward output shape."""
        batch, mel_dim, seq_len = 1, 80, 32
        x = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        mu = torch.randn(batch, mel_dim, seq_len)
        t = torch.tensor([0.5])
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out = dit(x, mask, mu, t, spks=spks, cond=cond)

        assert out.shape == (batch, mel_dim, seq_len)

    def test_timestep_embedding(self, dit):
        """Test that different timesteps produce different outputs."""
        batch, mel_dim, seq_len = 1, 80, 16
        x = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        mu = torch.randn(batch, mel_dim, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out1 = dit(x, mask, mu, torch.tensor([0.0]), spks=spks, cond=cond)
        out2 = dit(x, mask, mu, torch.tensor([1.0]), spks=spks, cond=cond)

        assert not torch.allclose(out1, out2)


class TestCFM:
    """Tests for Conditional Flow Matching classes."""

    @pytest.fixture
    def dummy_estimator(self):
        """Create a dummy estimator for testing."""

        class DummyEstimator(nn.Module):
            def __init__(self, mel_dim=80):
                super().__init__()
                self.mel_dim = mel_dim

            def forward(self, x, mask, mu, t, spks=None, cond=None):
                return torch.zeros_like(x)

        return DummyEstimator()

    def test_causal_conditional_cfm_forward(self, dummy_estimator):
        """Test CausalConditionalCFM forward pass."""
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        cfm_params = DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        )

        cfm = CausalConditionalCFM(
            in_channels=80,
            cfm_params=cfm_params,
            n_spks=1,
            spk_emb_dim=80,
            estimator=dummy_estimator,
        )

        batch, mel_dim, seq_len = 1, 80, 32
        mu = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond)

        assert out.shape == mu.shape


class TestSDPAFallback:
    """Test SDPA fallback for float32 inputs."""

    def test_float32_uses_sdpa(self):
        """Test that float32 inputs use SDPA fallback."""
        from vllm_omni.diffusion.attention.layer import Attention

        attn = Attention(
            num_heads=8,
            head_size=64,
            causal=False,
            softmax_scale=1.0 / 8.0,
        )

        batch, seq_len, heads, dim = 1, 16, 8, 64
        q = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)
        k = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)
        v = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)

        # Should not raise error - SDPA fallback handles float32
        out = attn(q, k, v)

        assert out.shape == (batch, seq_len, heads, dim)
        assert out.dtype == torch.float32
