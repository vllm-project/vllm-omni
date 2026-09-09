# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the model-local VibeVoice diffusion numerical kernel."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.vibevoice.diffusion import (
    VibeVoiceDiffusionGraphExecutor,
    VibeVoiceDiffusionSampler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DeterministicDiffusionHead(nn.Module):
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        latent_size = noisy_latents.shape[-1]
        return noisy_latents * 0.125 + condition[:, :latent_size] * 0.0625 + timesteps[:, None] * 1e-4


def _sampler() -> VibeVoiceDiffusionSampler:
    config = SimpleNamespace(
        ddpm_num_steps=1_000,
        ddpm_num_inference_steps=10,
        ddpm_beta_schedule="cosine",
        prediction_type="v_prediction",
        hidden_size=96,
        audio_config=SimpleNamespace(hidden_size=64),
    )
    return VibeVoiceDiffusionSampler.from_model_config(config)


def test_diffusion_graph_policy_has_only_four_official_control_keys() -> None:
    for batch_size in (1, 2, 3, 4):
        assert VibeVoiceDiffusionGraphExecutor.supports_graph_key(
            batch_size=batch_size,
            guidance_scale=1.3,
            num_inference_steps=10,
        )

    for batch_size, guidance_scale, num_inference_steps in (
        (0, 1.3, 10),
        (5, 1.3, 10),
        (1, 1.0, 10),
        (1, 2.5, 10),
        (1, 1.3, 5),
        (1, 1.3, 50),
    ):
        assert not VibeVoiceDiffusionGraphExecutor.supports_graph_key(
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = torch.linspace(-0.5, 0.5, 2 * 96).reshape(2, 96)
    negative = torch.linspace(0.25, -0.25, 2 * 96).reshape(2, 96)
    noise = torch.linspace(-1.0, 1.0, 4 * 64).reshape(4, 64)
    return positive, negative, noise


def _reference_sample(
    scheduler,
    head: nn.Module,
    positive: torch.Tensor,
    negative: torch.Tensor,
    noise: torch.Tensor,
    *,
    guidance_scale: float,
    num_inference_steps: int,
) -> torch.Tensor:
    batch_size = positive.shape[0]
    condition = torch.cat([positive, negative], dim=0)
    latent = noise.to(condition).clone()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    for timestep in scheduler.timesteps:
        combined = torch.cat([latent[:batch_size], latent[:batch_size]], dim=0)
        prediction = head(
            combined,
            timestep.repeat(combined.shape[0]).to(combined),
            condition,
        )
        conditional = prediction[:batch_size]
        unconditional = prediction[batch_size:]
        guided = unconditional + guidance_scale * (conditional - unconditional)
        latent = scheduler.step(
            torch.cat([guided, guided], dim=0),
            timestep,
            latent,
        ).prev_sample
    return latent[:batch_size].unsqueeze(1)


def test_diffusion_sampler_builds_normalized_fresh_schedulers() -> None:
    sampler = _sampler()
    assert sampler.beta_schedule == "squaredcos_cap_v2"
    assert sampler.prediction_type == "v_prediction"
    assert sampler.condition_size == 96
    assert sampler.latent_size == 64
    assert sampler.default_num_inference_steps == 10

    first = sampler.create_scheduler()
    second = sampler.create_scheduler()
    assert first is not second
    first.set_timesteps(num_inference_steps=10)
    assert first.num_inference_steps == 10
    assert second.num_inference_steps is None
    assert second.step_index is None


def test_diffusion_kernel_matches_an_independent_reference_loop() -> None:
    sampler = _sampler()
    head = _DeterministicDiffusionHead()
    positive, negative, noise = _inputs()
    original_noise = noise.clone()

    actual = sampler.sample_audio_latent(
        head,
        positive,
        negative,
        noise,
        guidance_scale=1.3,
        num_inference_steps=10,
    )
    expected = _reference_sample(
        sampler.create_scheduler(),
        head,
        positive,
        negative,
        noise,
        guidance_scale=1.3,
        num_inference_steps=10,
    )

    assert actual.shape == (2, 1, 64)
    assert torch.equal(actual, expected)
    assert torch.equal(noise, original_noise)
    assert torch.isfinite(actual).all()


def test_cached_scheduler_handles_alternating_step_counts() -> None:
    """Cached mutable scheduler state must reset between unlike requests."""
    sampler = _sampler()
    head = _DeterministicDiffusionHead()
    positive, negative, noise = _inputs()

    for steps in (10, 5, 10, 7, 5):
        actual = sampler.sample_audio_latent(
            head,
            positive,
            negative,
            noise,
            guidance_scale=1.3,
            num_inference_steps=steps,
        )
        expected = _reference_sample(
            sampler.create_scheduler(),
            head,
            positive,
            negative,
            noise,
            guidance_scale=1.3,
            num_inference_steps=steps,
        )
        assert torch.equal(actual, expected)


def test_forward_with_projected_condition_is_bitwise_identical() -> None:
    """cond_proj hoist used by the graph executor must not change values."""
    from vllm_omni.model_executor.models.vibevoice.diffusion import (
        VibeVoiceDiffusionHead,
    )

    config = SimpleNamespace(
        audio_config=SimpleNamespace(hidden_size=64),
        hidden_size=96,
        intermediate_size=128,
        hidden_act="silu",
        mlp_bias=False,
        rms_norm_eps=1e-6,
        num_head_layers=3,
        frequency_embedding_size=32,
        diffusion_max_period=10_000,
    )
    torch.manual_seed(0)
    head = VibeVoiceDiffusionHead(config).eval()
    noisy = torch.randn(4, 64)
    timesteps = torch.full((4,), 500.0)
    condition = torch.randn(4, 96)

    with torch.inference_mode():
        expected = head(noisy, timesteps, condition)
        actual = head.forward_with_projected_condition(
            noisy,
            timesteps,
            head.cond_proj(condition),
        )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("positive", "negative", "noise", "guidance_scale", "steps", "message"),
    [
        (
            torch.zeros(2, 96),
            torch.zeros(1, 96),
            torch.zeros(4, 64),
            1.3,
            10,
            "condition shapes must match",
        ),
        (
            torch.zeros(2, 96),
            torch.zeros(2, 96),
            torch.zeros(2, 64),
            1.3,
            10,
            "noise must preserve the official cond/uncond shape",
        ),
        (
            torch.zeros(2, 96),
            torch.zeros(2, 96),
            torch.zeros(4, 64),
            float("nan"),
            10,
            "guidance_scale must be finite",
        ),
        (
            torch.zeros(2, 96),
            torch.zeros(2, 96),
            torch.zeros(4, 64),
            1.3,
            0,
            "num_inference_steps must be positive",
        ),
    ],
)
def test_diffusion_kernel_rejects_invalid_contracts(
    positive: torch.Tensor,
    negative: torch.Tensor,
    noise: torch.Tensor,
    guidance_scale: float,
    steps: int,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _sampler().sample_audio_latent(
            _DeterministicDiffusionHead(),
            positive,
            negative,
            noise,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        )
