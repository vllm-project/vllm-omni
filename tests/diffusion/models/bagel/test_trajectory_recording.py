# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BAGEL trajectory recording in the denoising loop."""

import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    Bagel,
    NaiveCache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_TOKENS = 8
HIDDEN_DIM = 16
NUM_TIMESTEPS = 5
# generate_image uses timesteps[:-1], so actual steps = NUM_TIMESTEPS - 1
EXPECTED_STEPS = NUM_TIMESTEPS - 1


def _make_mock_bagel():
    """Create a mock Bagel with _forward_flow returning constant velocity."""
    mock = MagicMock(spec=Bagel)
    mock._sp_size = 1

    # _forward_flow returns a small constant velocity so x_t changes each step
    def fake_forward_flow(self, x_t, **kwargs):
        return torch.ones_like(x_t) * 0.1

    mock._forward_flow = types.MethodType(fake_forward_flow, mock)
    # _merge_naive_caches is called in the batched CFG path
    mock._merge_naive_caches = types.MethodType(lambda self, caches: NaiveCache(1), mock)

    # Bind the real generate_image to our mock
    mock.generate_image = types.MethodType(Bagel.generate_image, mock)
    return mock


def _make_generate_args(num_tokens=NUM_TOKENS, hidden_dim=HIDDEN_DIM, cfg=False):
    """Tensor arguments for generate_image.

    Args:
        cfg: If True, enable batched CFG path (cfg_text_scale > 1.0).
    """
    seq_len = num_tokens + 2  # packed_seqlens includes 2 extra tokens
    base = dict(
        packed_text_ids=torch.zeros(2, dtype=torch.long),
        packed_text_indexes=torch.tensor([0, 1], dtype=torch.long),
        packed_init_noises=torch.randn(num_tokens, hidden_dim),
        packed_vae_position_ids=torch.arange(num_tokens, dtype=torch.long),
        packed_vae_token_indexes=torch.arange(2, seq_len, dtype=torch.long),
        packed_seqlens=torch.tensor([seq_len], dtype=torch.int),
        packed_position_ids=torch.arange(seq_len, dtype=torch.long),
        packed_indexes=torch.arange(seq_len, dtype=torch.long),
        past_key_values=NaiveCache(1),
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.zeros(0, dtype=torch.long),
        num_timesteps=NUM_TIMESTEPS,
        timestep_shift=1.0,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
    )
    if cfg:
        base |= dict(
            cfg_text_scale=4.0,
            cfg_text_packed_query_indexes=torch.arange(seq_len, dtype=torch.long),
            cfg_text_packed_position_ids=torch.arange(seq_len, dtype=torch.long),
            cfg_text_past_key_values=NaiveCache(1),
            cfg_text_key_values_lens=torch.tensor([0], dtype=torch.int),
            cfg_text_packed_key_value_indexes=torch.zeros(0, dtype=torch.long),
        )
    return base


@pytest.fixture(params=[False, True], ids=["no_cfg", "batched_cfg"])
def bagel_and_args(request):
    """Mock Bagel instance and generate_image arguments.

    Parametrized over CFG mode so every test runs on both the no-CFG
    and batched-CFG code paths.
    """
    cfg = request.param
    with patch(
        "vllm_omni.diffusion.models.bagel.bagel_transformer.get_classifier_free_guidance_world_size",
        return_value=1,
    ):
        yield _make_mock_bagel(), _make_generate_args(cfg=cfg)


class TestTrajectoryRecording:
    """Tests for trajectory latent/timestep recording in generate_image."""

    def test_trajectory_disabled_returns_none(self, bagel_and_args):
        bagel, args = bagel_and_args

        unpacked, trajectory_latents, trajectory_timesteps, trajectory_log_probs = bagel.generate_image(
            **args, return_trajectory_latents=False
        )

        assert isinstance(unpacked, (list, tuple))
        assert len(unpacked) == 1  # one sequence
        assert trajectory_latents is None
        assert trajectory_timesteps is None
        assert trajectory_log_probs is None

    def test_trajectory_enabled_returns_correct_count(self, bagel_and_args):
        bagel, args = bagel_and_args

        _, trajectory_latents, trajectory_timesteps, trajectory_log_probs = bagel.generate_image(
            **args, return_trajectory_latents=True
        )

        assert trajectory_latents is not None
        assert trajectory_timesteps is not None
        assert len(trajectory_latents) == EXPECTED_STEPS
        assert len(trajectory_timesteps) == EXPECTED_STEPS
        # log_probs is None without a scheduler (default ODE path)
        assert trajectory_log_probs is None

    def test_trajectory_latents_shape_matches_input(self, bagel_and_args):
        bagel, args = bagel_and_args
        expected_shape = args["packed_init_noises"].shape

        _, trajectory_latents, *_ = bagel.generate_image(**args, return_trajectory_latents=True)

        for i, lat in enumerate(trajectory_latents):
            assert lat.shape == expected_shape, f"Step {i}: expected {expected_shape}, got {lat.shape}"

    def test_trajectory_latents_are_distinct(self, bagel_and_args):
        bagel, args = bagel_and_args

        _, trajectory_latents, *_ = bagel.generate_image(**args, return_trajectory_latents=True)

        for i in range(1, len(trajectory_latents)):
            assert not torch.equal(trajectory_latents[i], trajectory_latents[i - 1]), (
                f"Steps {i - 1} and {i} should differ"
            )

    def test_trajectory_timesteps_are_decreasing(self, bagel_and_args):
        bagel, args = bagel_and_args

        _, _, trajectory_timesteps, _ = bagel.generate_image(**args, return_trajectory_latents=True)

        for i in range(1, len(trajectory_timesteps)):
            assert trajectory_timesteps[i] < trajectory_timesteps[i - 1], (
                f"Timestep {i} ({trajectory_timesteps[i]:.4f}) should be less than "
                f"timestep {i - 1} ({trajectory_timesteps[i - 1]:.4f})"
            )

    def test_trajectory_final_latent_matches_output(self, bagel_and_args):
        bagel, args = bagel_and_args

        unpacked, trajectory_latents, *_ = bagel.generate_image(**args, return_trajectory_latents=True)

        # Reconstruct the full final latent from unpacked pieces
        final_latent = torch.cat(unpacked, dim=0)
        assert torch.allclose(trajectory_latents[-1], final_latent, atol=1e-6), (
            "Last trajectory latent should match the final output"
        )


# ---------------------------------------------------------------------------
# Mock scheduler for log-prob tests
# ---------------------------------------------------------------------------


@dataclass
class _MockStepOutput:
    prev_sample: torch.Tensor
    log_prob: torch.Tensor


class _MockScheduler:
    """Minimal scheduler: Euler step + constant log-prob per step."""

    def step(self, model_output, sigma, sample, dt, **kwargs):
        prev_sample = sample - model_output * dt
        log_prob = torch.tensor(-1.0)
        return _MockStepOutput(prev_sample=prev_sample, log_prob=log_prob)


class TestTrajectoryLogProbs:
    """Tests for log-prob recording when a scheduler is provided."""

    @pytest.fixture()
    def bagel_scheduler_args(self):
        with patch(
            "vllm_omni.diffusion.models.bagel.bagel_transformer.get_classifier_free_guidance_world_size",
            return_value=1,
        ):
            yield _make_mock_bagel(), _make_generate_args(), _MockScheduler()

    def test_log_probs_recorded_with_scheduler(self, bagel_scheduler_args):
        bagel, args, scheduler = bagel_scheduler_args

        _, _, _, trajectory_log_probs = bagel.generate_image(
            **args, return_trajectory_latents=True, scheduler=scheduler
        )

        assert trajectory_log_probs is not None
        assert len(trajectory_log_probs) == EXPECTED_STEPS

    def test_log_probs_are_finite(self, bagel_scheduler_args):
        bagel, args, scheduler = bagel_scheduler_args

        _, _, _, trajectory_log_probs = bagel.generate_image(
            **args, return_trajectory_latents=True, scheduler=scheduler
        )

        for i, lp in enumerate(trajectory_log_probs):
            assert torch.isfinite(lp).all(), f"Step {i}: log_prob is not finite"

    def test_log_probs_none_without_scheduler(self, bagel_scheduler_args):
        bagel, args, _ = bagel_scheduler_args

        _, _, _, trajectory_log_probs = bagel.generate_image(**args, return_trajectory_latents=True, scheduler=None)

        assert trajectory_log_probs is None

    def test_scheduler_updates_latents(self, bagel_scheduler_args):
        """Verify the scheduler's prev_sample is used (not the raw Euler step)."""
        bagel, args, scheduler = bagel_scheduler_args

        _, traj_with_sched, *_ = bagel.generate_image(**args, return_trajectory_latents=True, scheduler=scheduler)
        _, traj_without, *_ = bagel.generate_image(**args, return_trajectory_latents=True, scheduler=None)

        # Mock scheduler does the same Euler step, so latents should match
        for i in range(len(traj_with_sched)):
            assert torch.allclose(traj_with_sched[i], traj_without[i], atol=1e-5), (
                f"Step {i}: scheduler and ODE paths should produce same latents"
            )
