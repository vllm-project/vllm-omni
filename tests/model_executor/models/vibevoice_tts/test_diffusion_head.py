# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the VibeVoice diffusion head and DPM-Solver scheduler.

Verifies forward pass shapes, weight loading, and scheduler behavior.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.vibevoice_tts.configuration_vibevoice_tts import (
    VibeVoiceDiffusionHeadConfig,
)
from vllm_omni.model_executor.models.vibevoice_tts.vibevoice_tts_diffusion_head import (
    DPMSolverScheduler,
    VibeVoiceDiffusionHead,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def config():
    return VibeVoiceDiffusionHeadConfig(
        hidden_size=128,
        head_layers=2,
        head_ffn_ratio=2.0,
        rms_norm_eps=1e-5,
        latent_size=32,
        prediction_type="v_prediction",
        ddpm_num_steps=100,
        ddpm_num_inference_steps=10,
        ddpm_beta_schedule="cosine",
    )


@pytest.fixture
def head(config):
    return VibeVoiceDiffusionHead(config).eval()


# ──────────────────────────────────────────────────────────────────
# Diffusion Head forward
# ──────────────────────────────────────────────────────────────────


class TestDiffusionHeadForward:
    def test_output_shape(self, head, config):
        """Forward produces correct output shape."""
        B = 4
        noisy = torch.randn(B, config.latent_size)
        timesteps = torch.randint(0, config.ddpm_num_steps, (B,)).float()
        condition = torch.randn(B, config.hidden_size)

        with torch.no_grad():
            out = head(noisy, timesteps, condition)

        assert out.shape == (B, config.latent_size)

    def test_single_sample(self, head, config):
        """Works with batch_size=1."""
        noisy = torch.randn(1, config.latent_size)
        timesteps = torch.tensor([50.0])
        condition = torch.randn(1, config.hidden_size)

        with torch.no_grad():
            out = head(noisy, timesteps, condition)

        assert out.shape == (1, config.latent_size)

    def test_deterministic(self, head, config):
        """Same input produces same output."""
        B = 2
        noisy = torch.randn(B, config.latent_size)
        timesteps = torch.tensor([10.0, 50.0])
        condition = torch.randn(B, config.hidden_size)

        with torch.no_grad():
            out1 = head(noisy, timesteps, condition)
            out2 = head(noisy, timesteps, condition)

        torch.testing.assert_close(out1, out2)

    def test_different_timesteps_give_different_output(self, config):
        """Different timesteps produce different outputs (with non-zero modulation weights)."""
        head = VibeVoiceDiffusionHead(config).eval()
        # Override zero-initialized weights so timestep conditioning has an effect
        for layer in head.layers:
            nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.normal_(head.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.normal_(head.final_layer.linear.weight, std=0.02)

        noisy = torch.randn(1, config.latent_size)
        condition = torch.randn(1, config.hidden_size)

        with torch.no_grad():
            out_early = head(noisy, torch.tensor([1.0]), condition)
            out_late = head(noisy, torch.tensor([99.0]), condition)

        assert not torch.allclose(out_early, out_late)


class TestDiffusionHeadLoadWeight:
    def test_load_weight_known(self, head):
        """load_weight succeeds for a known parameter."""
        name = "noisy_images_proj.weight"
        w = torch.randn_like(head.noisy_images_proj.weight)
        loaded_name = head.load_weight((name, w))
        assert loaded_name == name
        torch.testing.assert_close(head.noisy_images_proj.weight.data, w)

    def test_load_weight_unknown(self, head):
        """load_weight for unknown name returns the name (logged warning)."""
        loaded_name = head.load_weight(("nonexistent.weight", torch.zeros(1)))
        assert loaded_name == "nonexistent.weight"


# ──────────────────────────────────────────────────────────────────
# DPM-Solver Scheduler
# ──────────────────────────────────────────────────────────────────


class TestDPMSolverScheduler:
    def test_set_timesteps(self):
        """set_timesteps creates descending timestep schedule."""
        sched = DPMSolverScheduler(num_train_timesteps=1000)
        sched.set_timesteps(20)
        assert sched.timesteps is not None
        assert len(sched.timesteps) == 20
        # Should be descending (reverse order)
        for i in range(len(sched.timesteps) - 1):
            assert sched.timesteps[i] >= sched.timesteps[i + 1]

    def test_add_noise_shape(self):
        """add_noise preserves tensor shape."""
        sched = DPMSolverScheduler(num_train_timesteps=100)
        original = torch.randn(4, 64)
        noise = torch.randn(4, 64)
        timesteps = torch.tensor([10, 20, 30, 40])
        noisy = sched.add_noise(original, noise, timesteps)
        assert noisy.shape == original.shape

    def test_add_noise_zero_timestep_close_to_original(self):
        """At t=0, noisy sample should be very close to original."""
        sched = DPMSolverScheduler(num_train_timesteps=1000)
        original = torch.randn(2, 32)
        noise = torch.randn(2, 32)
        timesteps = torch.tensor([0, 0])
        noisy = sched.add_noise(original, noise, timesteps)
        # alpha_cumprod[0] should be ~1.0, so noisy ≈ original
        assert torch.allclose(noisy, original, atol=0.05)

    def test_get_velocity_shape(self):
        """get_velocity preserves tensor shape."""
        sched = DPMSolverScheduler(num_train_timesteps=100)
        sample = torch.randn(4, 64)
        noise = torch.randn(4, 64)
        timesteps = torch.tensor([10, 20, 30, 40])
        vel = sched.get_velocity(sample, noise, timesteps)
        assert vel.shape == sample.shape

    def test_step_returns_tensor(self):
        """step() returns a tensor of the same shape."""
        sched = DPMSolverScheduler(
            num_train_timesteps=100,
            prediction_type="v_prediction",
        )
        sched.set_timesteps(10)
        model_output = torch.randn(4, 32)
        sample = torch.randn(4, 32)
        t = sched.timesteps[0]
        prev = sched.step(model_output, t, sample)
        assert isinstance(prev, torch.Tensor)
        assert prev.shape == sample.shape

    def test_full_denoise_loop(self):
        """Full denoise loop reduces noise (output differs from pure noise)."""
        sched = DPMSolverScheduler(
            num_train_timesteps=100,
            prediction_type="v_prediction",
            beta_schedule="cosine",
        )
        sched.set_timesteps(10)

        torch.manual_seed(42)
        # Start from noise
        sample = torch.randn(2, 32)
        initial_norm = sample.norm()

        # Dummy "model" that predicts zeros
        for t in sched.timesteps:
            model_output = torch.zeros_like(sample)
            sample = sched.step(model_output, t, sample)

        # Should have changed from the initial noise
        assert not torch.allclose(sample, torch.randn(2, 32))

    def test_cosine_schedule_bounds(self):
        """Cosine schedule produces betas in (0, 1)."""
        sched = DPMSolverScheduler(
            num_train_timesteps=1000,
            beta_schedule="cosine",
        )
        assert torch.all(sched.alphas_cumprod > 0)
        assert torch.all(sched.alphas_cumprod <= 1)
        # alpha_cumprod should be monotonically decreasing
        diff = sched.alphas_cumprod[1:] - sched.alphas_cumprod[:-1]
        assert torch.all(diff <= 0)

    def test_unknown_prediction_type_raises(self):
        """Unknown prediction_type in step() raises ValueError."""
        sched = DPMSolverScheduler(
            num_train_timesteps=100,
            prediction_type="unknown_type",
        )
        sched.set_timesteps(5)
        with pytest.raises(ValueError, match="Unknown prediction_type"):
            sched.step(torch.zeros(1, 4), sched.timesteps[0], torch.zeros(1, 4))
