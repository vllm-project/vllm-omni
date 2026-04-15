# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for Wan2.2 SupportsStepExecution protocol implementation.

Tests use lightweight mocks (no real model weights) to verify:
- Protocol compliance (class flag, method presence)
- Helper correctness (_resolve_generation_params, _select_model_for_timestep)
- Step execution decomposition matches monolithic forward()
- I2V mode latent input preparation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from vllm_omni.diffusion.worker.utils import DiffusionRequestState

# ---------------------------------------------------------------------------
# Shared test utilities
# ---------------------------------------------------------------------------


def _make_sampling(**overrides):
    """Create a mock sampling params object."""
    sampling = MagicMock()
    sampling.height = overrides.get("height", 480)
    sampling.width = overrides.get("width", 832)
    sampling.num_frames = overrides.get("num_frames", 81)
    sampling.num_inference_steps = overrides.get("num_inference_steps", 4)
    sampling.guidance_scale = overrides.get("guidance_scale", 1.0)
    sampling.guidance_scale_provided = overrides.get("guidance_scale_provided", True)
    sampling.guidance_scale_2 = overrides.get("guidance_scale_2", None)
    sampling.boundary_ratio = overrides.get("boundary_ratio", None)
    sampling.num_outputs_per_prompt = overrides.get("num_outputs_per_prompt", 1)
    sampling.max_sequence_length = overrides.get("max_sequence_length", 512)
    sampling.seed = overrides.get("seed", 42)
    sampling.generator = None
    sampling.latents = None
    return sampling


def _make_state(**overrides):
    """Create a DiffusionRequestState with mock sampling."""
    return DiffusionRequestState(
        req_id="test",
        sampling=_make_sampling(**overrides),
        prompts=overrides.get("prompts", ["test prompt"]),
    )


def _make_pipeline_stub():
    """Create a minimal Wan22Pipeline without __init__ (no model weights)."""
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor_spatial = 8
    pipeline.vae_scale_factor_temporal = 4
    pipeline.boundary_ratio = 0.875
    pipeline.expand_timesteps = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
    pipeline._current_timestep = None
    pipeline._pp_send_work_list = []

    config = MagicMock()
    config.patch_size = (1, 2, 2)
    config.in_channels = 16
    config.out_channels = 16
    pipeline.transformer_config = config

    pipeline.device = torch.device("cpu")

    return pipeline


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


class TestWan22SupportsStepExecution:
    """Verify the class-level protocol flag and method signatures."""

    def test_class_var_is_true(self):
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

        assert hasattr(Wan22Pipeline, "supports_step_execution")
        assert Wan22Pipeline.supports_step_execution is True

    def test_has_required_methods(self):
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

        for method_name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
            assert hasattr(Wan22Pipeline, method_name), f"Missing method: {method_name}"

# ---------------------------------------------------------------------------
# 2. _resolve_generation_params helper
# ---------------------------------------------------------------------------


class TestResolveGenerationParams:
    """Verify parameter resolution and alignment logic."""

    def test_dimensions_aligned_to_mod_value(self):
        pipeline = _make_pipeline_stub()
        pipeline.transformer = MagicMock(dtype=torch.bfloat16)
        pipeline.transformer_2 = None

        for height in range(400, 600, 50):
            for width in range(700, 900, 50):
                state = _make_state(height=height, width=width)
                params = pipeline._resolve_generation_params(state)
                # mod_value = 8 * 2 = 16
                assert params["height"] % 16 == 0
                assert params["width"] % 16 == 0

    def test_num_frames_aligned_to_vae_temporal(self):
        pipeline = _make_pipeline_stub()
        pipeline.transformer = MagicMock(dtype=torch.bfloat16)
        pipeline.transformer_2 = None

        for num_frames in range(1, 200):        
            state = _make_state(num_frames=num_frames)
            params = pipeline._resolve_generation_params(state)
            assert params["num_frames"] % pipeline.vae_scale_factor_temporal == 1 or params["num_frames"] == 1



# ---------------------------------------------------------------------------
# 3. _select_model_for_timestep helper
# ---------------------------------------------------------------------------


class TestSelectModelForTimestep:
    def _make(self):
        pipeline = _make_pipeline_stub()
        pipeline.transformer = MagicMock(name="transformer")
        pipeline.transformer_2 = MagicMock(name="transformer_2")
        pipeline._guidance_scale = 4.0
        pipeline._guidance_scale_2 = 7.0
        return pipeline

    def test_high_noise_uses_transformer(self):
        pipeline = self._make()
        model, scale = pipeline._select_model_for_timestep(torch.tensor(800.0), boundary_timestep=500.0)
        assert model is pipeline.transformer
        assert scale == 4.0

    def test_low_noise_uses_transformer_2(self):
        pipeline = self._make()
        model, scale = pipeline._select_model_for_timestep(torch.tensor(300.0), boundary_timestep=500.0)
        assert model is pipeline.transformer_2
        assert scale == 7.0

    def test_no_boundary_uses_transformer(self):
        pipeline = self._make()
        model, _ = pipeline._select_model_for_timestep(torch.tensor(300.0), boundary_timestep=None)
        assert model is pipeline.transformer

    def test_fallback_when_transformer_none(self):
        pipeline = self._make()
        pipeline.transformer = None
        model, _ = pipeline._select_model_for_timestep(torch.tensor(800.0), boundary_timestep=500.0)
        assert model is pipeline.transformer_2


# ---------------------------------------------------------------------------
# 4.1  Step execution decomposition matches forward()
# ---------------------------------------------------------------------------


class _FakeScheduler:
    """Minimal scheduler that applies a deterministic update: latents -= 0.1 * noise_pred."""

    def __init__(self, timesteps: torch.Tensor):
        self.timesteps = timesteps
        self._step_index = 0
        self.config = MagicMock()
        self.config.num_train_timesteps = 1000

    def set_timesteps(self, _num_steps, device=None):
        pass  # timesteps already set

    def step(self, noise_pred, _t, latents, return_dict=False):
        self._step_index += 1
        return (latents - 0.1 * noise_pred,)


class _FakeTransformer(torch.nn.Module):
    """Deterministic transformer: output = input * 0.5 (applied to hidden_states)."""

    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.tensor(1.0))

    @property
    def dtype(self):
        return torch.float32

    def forward(self, hidden_states, timestep, encoder_hidden_states, intermediate_tensors=None, **kwargs):
        # Simulate noise prediction: scale down hidden_states
        noise_pred = hidden_states * 0.5
        return (noise_pred,)


def _patch_parallel_state():
    """Context manager that patches PP and CFG parallel state to single-GPU (world_size=1)."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("vllm_omni.diffusion.distributed.pp_parallel.get_pipeline_parallel_world_size", return_value=1)
    )
    stack.enter_context(
        patch("vllm_omni.diffusion.distributed.cfg_parallel.get_classifier_free_guidance_world_size", return_value=1)
    )
    return stack


class TestDenoiseStepCorrectness:
    """Verify prepare_encode → denoise_step x N → step_scheduler x N → post_decode
    produces the same latent trajectory as running the equivalent loop manually."""

    def _make_pipeline(self):
        pipeline = _make_pipeline_stub()
        pipeline.transformer = _FakeTransformer()
        pipeline.transformer_2 = None
        pipeline.expand_timesteps = False

        timesteps = torch.tensor([900.0, 700.0, 500.0, 300.0])
        pipeline.scheduler = _FakeScheduler(timesteps)

        # Mock encode_prompt to return fixed embeddings
        prompt_embeds = torch.randn(1, 10, 64)
        pipeline.encode_prompt = MagicMock(return_value=(prompt_embeds, None))

        # Mock VAE decode (identity)
        vae = MagicMock()
        vae.dtype = torch.float32
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.config.z_dim = 16
        vae.decode = MagicMock(side_effect=lambda x, **kw: (x,))
        pipeline.vae = vae

        # Mock prepare_latents to return seeded noise
        torch.manual_seed(123)
        fixed_latents = torch.randn(1, 16, 21, 30, 52)
        pipeline.prepare_latents = MagicMock(return_value=fixed_latents.clone())

        return pipeline, fixed_latents.clone(), prompt_embeds

    def test_latent_trajectory_matches(self):
        """Step-by-step execution produces the same final latents as a manual loop."""
        pipeline, initial_latents, prompt_embeds = self._make_pipeline()
        timesteps = pipeline.scheduler.timesteps

        # ── Manual baseline loop ──
        latents = initial_latents.clone()
        for t in timesteps:
            latent_input = latents.to(torch.float32)
            noise_pred = pipeline.transformer(
                hidden_states=latent_input,
                timestep=t.expand(1),
                encoder_hidden_states=prompt_embeds,
            )[0]
            latents = latents - 0.1 * noise_pred
        baseline_latents = latents

        # ── Step execution path ──
        state = _make_state(num_inference_steps=4)
        state = pipeline.prepare_encode(state)

        with _patch_parallel_state():
            while not state.denoise_completed:
                noise_pred = pipeline.denoise_step(state)
                pipeline.step_scheduler(state, noise_pred)

        assert state.step_index == len(timesteps)
        torch.testing.assert_close(
            state.latents,
            baseline_latents,
            rtol=1e-5,
            atol=1e-5,
            msg="Step execution latents diverged from manual baseline",
        )

    def test_post_decode_calls_vae(self):
        """post_decode invokes VAE decode and returns DiffusionOutput."""
        from vllm_omni.diffusion.data import DiffusionOutput

        pipeline, _, _ = self._make_pipeline()

        state = _make_state(num_inference_steps=4)
        state = pipeline.prepare_encode(state)
        with _patch_parallel_state():
            while not state.denoise_completed:
                noise_pred = pipeline.denoise_step(state)
                pipeline.step_scheduler(state, noise_pred)

        mock_platform = MagicMock()
        mock_platform.is_available.return_value = False
        with (
            patch.object(type(pipeline), "sync_pp_send"),
            patch("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.current_omni_platform", mock_platform),
        ):
            result = pipeline.post_decode(state)

        assert isinstance(result, DiffusionOutput)
        assert result.output is not None
        pipeline.vae.decode.assert_called_once()

    def test_step_count_matches_timesteps(self):
        """Exactly len(timesteps) steps are executed."""
        pipeline, _, _ = self._make_pipeline()
        state = _make_state(num_inference_steps=4)
        state = pipeline.prepare_encode(state)

        step_count = 0
        with _patch_parallel_state():
            while not state.denoise_completed:
                noise_pred = pipeline.denoise_step(state)
                pipeline.step_scheduler(state, noise_pred)
                step_count += 1

        assert step_count == 4

    def test_scheduler_is_deepcopied(self):
        """Each request gets its own scheduler copy, not a shared reference."""
        pipeline, _, _ = self._make_pipeline()
        original_scheduler = pipeline.scheduler
        state = _make_state(num_inference_steps=4)
        state = pipeline.prepare_encode(state)
        assert state.scheduler is not original_scheduler


# ---------------------------------------------------------------------------
# 4.3  _prepare_latent_input I2V mode
# ---------------------------------------------------------------------------


class TestPrepareLatentInputI2V:
    """Verify I2V mode latent blending and timestep expansion."""

    def _make_pipeline_and_state(self):
        pipeline = _make_pipeline_stub()
        pipeline.transformer = MagicMock(dtype=torch.float32)
        pipeline.transformer_2 = None

        # Latents: [B=1, C=16, T=5, H=8, W=10]
        latents = torch.randn(1, 16, 5, 8, 10)
        # Condition: same shape, different values
        latent_condition = torch.randn(1, 16, 5, 8, 10)
        # Mask: 0 for first frame, 1 for rest
        first_frame_mask = torch.ones(1, 1, 5, 8, 10)
        first_frame_mask[:, :, 0] = 0

        state = _make_state()
        state.latents = latents
        state.extra["expand_timesteps"] = True
        state.extra["latent_condition"] = latent_condition
        state.extra["first_frame_mask"] = first_frame_mask

        return pipeline, state, latents, latent_condition, first_frame_mask

    def test_i2v_blends_condition_with_latents(self):
        """First frame uses condition, remaining frames use latents."""
        pipeline, state, latents, condition, mask = self._make_pipeline_and_state()
        t = torch.tensor(500.0)

        latent_input, _ = pipeline._prepare_latent_input(state, t, torch.float32)

        # First frame (mask=0): should be condition
        expected_first = condition[:, :, 0]
        torch.testing.assert_close(latent_input[:, :, 0], expected_first, rtol=1e-5, atol=1e-5)

        # Remaining frames (mask=1): should be latents
        expected_rest = latents[:, :, 1:]
        torch.testing.assert_close(latent_input[:, :, 1:], expected_rest, rtol=1e-5, atol=1e-5)

    def test_i2v_timestep_expansion(self):
        """Timestep is expanded per-patch: 0 for condition patches, t for noise patches."""
        pipeline, state, _, _, mask = self._make_pipeline_and_state()
        t = torch.tensor(500.0)

        _, timestep_tensor = pipeline._prepare_latent_input(state, t, torch.float32)

        # patch_size = (1, 2, 2) → patch dims: T=5, H=4, W=5
        # Sequence length = 5 * 4 * 5 = 100
        assert timestep_tensor.shape[0] == 1  # batch
        assert timestep_tensor.shape[1] == 5 * 4 * 5  # flattened patch sequence

        # First frame patches (first 4*5=20) should have timestep 0
        first_frame_patches = timestep_tensor[0, : 4 * 5]
        assert (first_frame_patches == 0).all(), "First frame patches should have timestep 0"

        # Remaining patches should have timestep = 500
        rest_patches = timestep_tensor[0, 4 * 5 :]
        assert (rest_patches == 500.0).all(), "Non-first-frame patches should have timestep t"

    def test_t2v_mode_passthrough(self):
        """T2V mode: latents pass through unchanged, timestep is broadcast."""
        pipeline = _make_pipeline_stub()
        pipeline.transformer = MagicMock(dtype=torch.float32)
        pipeline.transformer_2 = None

        latents = torch.randn(2, 16, 5, 8, 10)
        state = _make_state()
        state.latents = latents
        # No I2V extras → T2V mode

        t = torch.tensor(500.0)
        latent_input, timestep_tensor = pipeline._prepare_latent_input(state, t, torch.float32)

        torch.testing.assert_close(latent_input, latents)
        assert timestep_tensor.shape == (2,)
        assert (timestep_tensor == 500.0).all()


# ---------------------------------------------------------------------------
# 5. denoise_step vs Wan22Pipeline.forward()
# ---------------------------------------------------------------------------


def _make_req(num_inference_steps: int = 4):
    """Create a minimal T2V OmniDiffusionRequest for forward() comparison tests."""
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        max_sequence_length=512,
        num_outputs_per_prompt=1,
        seed=42,
    )
    return OmniDiffusionRequest(prompts=["test prompt"], sampling_params=sampling)


class TestDenoiseStepMatchesForward:
    """Verify denoise_step matches Wan22Pipeline.forward()."""

    def _make_pipeline(self):
        pipeline = _make_pipeline_stub()
        pipeline.transformer = _FakeTransformer()
        pipeline.transformer_2 = None
        pipeline.expand_timesteps = False

        timesteps = torch.tensor([900.0, 700.0, 500.0, 300.0])
        pipeline.scheduler = _FakeScheduler(timesteps)

        torch.manual_seed(17)
        prompt_embeds = torch.randn(1, 10, 64)
        pipeline.encode_prompt = MagicMock(return_value=(prompt_embeds, None))

        vae = MagicMock()
        vae.dtype = torch.float32
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.config.z_dim = 16
        pipeline.vae = vae

        torch.manual_seed(13)
        fixed_latents = torch.randn(1, 16, 21, 30, 52)
        pipeline.prepare_latents = MagicMock(return_value=fixed_latents.clone())

        return pipeline

    def _run_forward(self, pipeline):
        """Run pipeline.forward() in T2V mode and return the final latents."""
        req = _make_req(num_inference_steps=4)
        mock_platform = MagicMock()
        mock_platform.is_available.return_value = False
        with (
            patch("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.current_omni_platform", mock_platform),
            _patch_parallel_state(),
        ):
            result = pipeline.forward(req, output_type="latent")
        return result.output

    def test_denoise_step_matches_forward(self):
        """Full step-execution loop (prepare_encode → denoise_step x N → step_scheduler x N)
        produces the same final latents as Wan22Pipeline.forward()."""
        pipeline = self._make_pipeline()

        # Reference: monolithic forward()
        fwd_latents = self._run_forward(pipeline)

        # Step-execution path using the same pipeline (same latent / prompt_embed mocks)
        state = _make_state(num_inference_steps=4)
        state = pipeline.prepare_encode(state)
        with _patch_parallel_state():
            while not state.denoise_completed:
                noise_pred = pipeline.denoise_step(state)
                pipeline.step_scheduler(state, noise_pred)

        torch.testing.assert_close(state.latents, fwd_latents, rtol=1e-5, atol=1e-5)

