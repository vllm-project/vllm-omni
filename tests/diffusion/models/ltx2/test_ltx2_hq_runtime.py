# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime integration tests for the LTX-2.3/2.5 HQ sampler path."""

import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.models.ltx2.ltx2_denoise import prepare_scheduler_stage, run_res2s_phase
from vllm_omni.diffusion.models.ltx2.ltx2_guidance import LTXGuidanceExecutor, LTXGuidancePlan, LTXGuidanceSpec
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.ltx2_res2s import build_ltx2_res2s_sigmas

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_guidance_prediction_uses_explicit_sigma_instead_of_scheduler_index():
    video_state = torch.tensor([[[1.0, 2.0]]])
    audio_state = torch.tensor([[[3.0, 4.0]]])
    video_velocity = torch.tensor([[[2.0, -1.0]]])
    audio_velocity = torch.tensor([[[1.5, -0.5]]])
    context = torch.zeros(1, 1, 1)
    prompt = SimpleNamespace(
        positive_connector_prompt_embeds=context,
        positive_connector_audio_prompt_embeds=context,
        negative_connector_prompt_embeds=None,
        negative_connector_audio_prompt_embeds=None,
    )

    pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.95])),
        _build_transformer_kwargs=lambda *_args, **_kwargs: {},
        _transformer_cache_context=lambda *_args, **_kwargs: nullcontext(),
        transformer=lambda **_kwargs: (video_velocity, audio_velocity),
        _video_guidance_model_sigma=lambda sigma, _ctx: sigma,
    )
    forward_ctx = SimpleNamespace(
        guidance_parallel_ready=False,
        prompt_context=prompt,
        attention_kwargs=None,
        original_audio_num_frames=1,
    )
    denoise_ctx = SimpleNamespace()
    state = LTXAVState(video=video_state, audio=audio_state)

    video_x0, audio_x0 = LTXGuidanceExecutor().predict_denoised(
        pipeline,
        LTXGuidancePlan.build(LTXGuidanceSpec.positive_only()),
        0,
        torch.tensor(500.0),
        state,
        forward_ctx,
        denoise_ctx,
        video_sigma=torch.tensor(0.2),
        audio_sigma=torch.tensor(0.3),
    )

    torch.testing.assert_close(video_x0, video_state - video_velocity * 0.2)
    torch.testing.assert_close(audio_x0, audio_state - audio_velocity * 0.3)


def test_latent_dependent_scheduler_stage_uses_actual_packed_token_count():
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        shift_terminal=0.1,
        base_image_seq_len=1024,
        max_image_seq_len=4096,
    )
    pipeline = SimpleNamespace(scheduler=scheduler, device=torch.device("cpu"))
    token_count = 16 * 17 * 30

    _, _, timesteps = prepare_scheduler_stage(
        pipeline,
        SimpleNamespace(num_inference_steps=15, generator=None),
        device=torch.device("cpu"),
        sigmas=None,
        timesteps=None,
        latent_num_frames=16,
        latent_height=17,
        latent_width=30,
        video_token_count=token_count,
        use_official_sigma_schedule=False,
        use_latent_dependent_sigma_schedule=True,
        sampler="res2s",
    )

    expected = build_ltx2_res2s_sigmas(15, token_count)
    torch.testing.assert_close(pipeline.scheduler.sigmas, expected, rtol=0, atol=0)
    torch.testing.assert_close(timesteps, expected[:-1] * 1000, rtol=0, atol=0)


def test_res2s_phase_threads_midpoint_sigma_and_restores_i2v_tokens():
    model_calls: list[tuple[float, float, float]] = []
    model_inputs: list[torch.Tensor] = []
    model_audio_inputs: list[torch.Tensor] = []
    progress_updates = 0

    def predict_denoised(
        _index,
        timestep,
        state,
        _forward_ctx,
        _denoise_ctx,
        *,
        video_sigma,
        audio_sigma,
    ):
        model_calls.append((float(timestep), float(video_sigma), float(audio_sigma)))
        model_inputs.append(state.video.clone())
        model_audio_inputs.append(state.audio.clone())
        return torch.zeros_like(state.video), torch.ones_like(state.audio)

    @contextmanager
    def progress_bar(*, total):
        assert total == 2

        class Progress:
            def update(self):
                nonlocal progress_updates
                progress_updates += 1

        yield Progress()

    pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(
            sigmas=torch.tensor([0.8, 0.4, 0.0]),
            config={"num_train_timesteps": 1000},
        ),
        _predict_denoised_for_step=predict_denoised,
        progress_bar=progress_bar,
    )
    clean_video = torch.tensor([[[7.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    conditioning_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    state = LTXAVState(
        video=clean_video.clone(),
        audio=torch.zeros(1, 2, 2, dtype=torch.float64),
    )
    forward_ctx = SimpleNamespace(original_audio_num_frames=1)
    denoise_ctx = SimpleNamespace(
        latents=state.video,
        audio_latents=state.audio,
        clean_video_latents=clean_video,
        conditioning_mask=conditioning_mask,
    )

    result = run_res2s_phase(pipeline, state, forward_ctx, denoise_ctx)

    expected_sigmas = [0.8, math.sqrt(0.8 * 0.4), 0.4, math.sqrt(0.4 * 0.0011), 0.0011]
    assert [call[1] for call in model_calls] == pytest.approx(expected_sigmas)
    assert [call[2] for call in model_calls] == pytest.approx(expected_sigmas)
    assert [call[0] for call in model_calls] == pytest.approx([sigma * 1000 for sigma in expected_sigmas])
    assert progress_updates == 2
    for model_input in model_inputs:
        assert model_input[0, 0, 0] == 7.0
    for model_audio_input in model_audio_inputs:
        torch.testing.assert_close(model_audio_input[:, 1:], torch.zeros_like(model_audio_input[:, 1:]))
    assert result.video[0, 0, 0] == 7.0
    torch.testing.assert_close(result.audio[:, 1:], torch.zeros_like(result.audio[:, 1:]))
