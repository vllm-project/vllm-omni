# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock, patch

import pytest
import torch

import vllm_omni.diffusion.models.magi2.sampler_magi2 as sampler_module
import vllm_omni.diffusion.profiler.diffusion_pipeline_profiler as profiler_module
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import Magi2Pipeline
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig, Magi2PreviewSampler, SamplerInput

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


class _Scheduler:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def step(self, noise, timestep, latent, *, return_dict):
        assert return_dict is False
        self.events.append((self.name, timestep.detach().clone()))
        return (latent.add_(noise * -0.05) + torch.randn_like(latent) * 0.001,)


def _request(device, dtype, mode="plain"):
    events: list[tuple[str, torch.Tensor]] = []
    config = CFGConfig(
        video_txt_guidance_scale=5.0,
        audio_txt_guidance_scale=3.0,
        use_dynamic_cfg=mode == "advanced",
        use_cfg_trick=mode == "advanced",
        cfg_trick_start_frame=1,
        cfg_rescale=0.2 if mode == "advanced" else 0.0,
        use_skimmed_cfg_linear=mode == "advanced",
    )
    generator = torch.Generator().manual_seed(31)

    def tensor(*shape):
        return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)

    request = SamplerInput(
        cfg_config=config,
        video_scheduler=_Scheduler("video", events),
        audio_scheduler=_Scheduler("audio", events),
        latent=tensor(1, 2, 2, 2, 2),
        audio_latent=tensor(1, 4, 2),
        txt_feat=tensor(1, 3, 4),
        null_txt_feat=tensor(1, 3, 4),
        ref_audio_feat=None,
        ref_video_feat=None,
        ref_image_feat=None,
        ref_image_feat_len=None,
        ref_image_special_token_embedding=None,
        video_t_list=torch.tensor([1000.0, 700.0, 100.0], device=device),
        # Existing semantics advance both schedulers on the video timeline.
        audio_t_list=torch.tensor([999.0, 500.0, 0.0], device=device),
    )
    return request, events


def _sampler():
    sampler = Magi2PreviewSampler(model=Mock(), data_proxy=Mock())

    def forward(model_input):
        batch = model_input.x_t.shape[0]
        text = model_input.txt_feat.mean(dim=(1, 2))
        video = model_input.x_t * 0.25 + text.view(batch, 1, 1, 1, 1)
        audio = model_input.audio_x_t * 0.25 + text.view(batch, 1, 1)
        return video + torch.randn_like(video) * 0.002, audio + torch.randn_like(audio) * 0.002

    def cfg_dispatch(**kwargs):
        assert kwargs["do_true_cfg"] and not kwargs["cfg_normalize"]
        assert kwargs["true_cfg_scale"] == 1.0
        return sampler.combine_cfg_noise(
            forward(kwargs["positive_kwargs"]["model_input"]),
            forward(kwargs["negative_kwargs"]["model_input"]),
            kwargs["true_cfg_scale"],
            cfg_normalize=kwargs["cfg_normalize"],
            kwargs=kwargs["kwargs"],
        )

    sampler.forward = Mock(side_effect=forward)
    sampler.predict_noise_maybe_with_cfg = Mock(side_effect=cfg_dispatch)
    return sampler


@torch.inference_mode()
def _inline_reference(sampler, request: SamplerInput, cfg_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    video_timesteps = list(request.video_t_list)
    audio_timesteps = list(request.audio_t_list)
    if not video_timesteps:
        raise ValueError("MAGI-2 sampling needs at least one timestep")
    if len(video_timesteps) != len(audio_timesteps):
        raise ValueError("video and audio timestep schedules must have equal lengths")

    video_cfgs, audio_cfgs = sampler.precalculate_cfg(
        video_timesteps,
        request.latent.shape[2],
        request.cfg_config,
        device=request.latent.device,
    )
    latent = request.latent.clone()
    audio_latent = request.audio_latent.clone()

    for timestep, video_cfg, audio_cfg in zip(
        video_timesteps,
        video_cfgs,
        audio_cfgs,
        strict=True,
    ):
        model_input = sampler.prepare_model_input(
            latent=latent,
            audio_latent=audio_latent,
            txt_feat=request.txt_feat,
            null_txt_feat=request.null_txt_feat,
            ref_audio_feat=request.ref_audio_feat,
            ref_video_feat=request.ref_video_feat,
            ref_image_feat=request.ref_image_feat,
            ref_image_feat_len=request.ref_image_feat_len,
            ref_image_special_token_embedding=(request.ref_image_special_token_embedding),
            t=timestep,
            cfg_config=request.cfg_config,
        )
        if cfg_size > 1:
            positive_input, negative_input = sampler._split_cfg_model_input(model_input)
            guided = sampler.predict_noise_maybe_with_cfg(
                do_true_cfg=True,
                true_cfg_scale=1.0,
                positive_kwargs={"model_input": positive_input},
                negative_kwargs={"model_input": negative_input},
                cfg_normalize=False,
                kwargs={
                    "video_txt_guidance_scale": video_cfg,
                    "audio_txt_guidance_scale": audio_cfg,
                    "cfg_config": request.cfg_config,
                    "latent": latent,
                    "audio_latent": audio_latent,
                },
            )
            if not isinstance(guided, tuple) or len(guided) != 2:
                raise RuntimeError("MAGI-2 CFG parallel combine must return video and audio predictions")
            latent, audio_latent = sampler._step_guided(
                guided,
                latent,
                audio_latent,
                request.video_scheduler,
                request.audio_scheduler,
                timestep,
            )
        else:
            model_pred = sampler.forward(model_input)
            latent, audio_latent, _, _ = sampler.step(
                model_pred,
                latent,
                audio_latent,
                video_cfg,
                audio_cfg,
                request.video_scheduler,
                request.audio_scheduler,
                timestep,
                cfg_config=request.cfg_config,
            )

    return latent, audio_latent


def _seed_and_state(device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        if device == "musa":
            torch.musa.manual_seed_all(seed)
    return torch.musa.get_rng_state() if device == "musa" else torch.get_rng_state()


def _check_equivalence(device, dtype, cfg_size, mode):
    expected_input, expected_events = _request(device, dtype, mode)
    actual_input, actual_events = _request(device, dtype, mode)
    expected_sampler, actual_sampler = _sampler(), _sampler()
    input_latents = (actual_input.latent.clone(), actual_input.audio_latent.clone())
    _seed_and_state(device, 82)
    expected = _inline_reference(expected_sampler, expected_input, cfg_size)
    expected_rng = _seed_and_state(device)
    _seed_and_state(device, 82)
    with patch.object(sampler_module, "get_classifier_free_guidance_world_size", return_value=cfg_size):
        with patch.object(actual_sampler, "denoise_step", wraps=actual_sampler.denoise_step) as steps:
            actual = actual_sampler.sample(actual_input)
    assert steps.call_count == 3
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        assert not output.requires_grad
    assert torch.equal(_seed_and_state(device), expected_rng)
    torch.testing.assert_close(actual_input.latent, input_latents[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_input.audio_latent, input_latents[1], rtol=0, atol=0)
    assert [name for name, _ in actual_events] == ["video", "audio"] * 3
    torch.testing.assert_close(
        torch.stack([t for _, t in actual_events]), torch.stack([t for _, t in expected_events]), rtol=0, atol=0
    )
    assert actual_sampler.forward.call_count == (3 if cfg_size == 1 else 0)
    assert actual_sampler.predict_noise_maybe_with_cfg.call_count == (0 if cfg_size == 1 else 3)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cfg_size", [1, 2])
@pytest.mark.parametrize("mode", ["plain", "advanced"])
def test_step_extraction_preserves_outputs_rng_and_scheduler_order(dtype, cfg_size, mode):
    _check_equivalence("cpu", dtype, cfg_size, mode)


@pytest.mark.cpu
def test_empty_or_mismatched_schedules_do_not_run_a_step():
    request, _ = _request("cpu", torch.float32)
    sampler = _sampler()
    sampler.denoise_step = Mock()
    request.audio_t_list = []
    with pytest.raises(ValueError, match="equal lengths"):
        sampler.sample(request)
    request.video_t_list = []
    with pytest.raises(ValueError, match="at least one"):
        sampler.sample(request)
    sampler.denoise_step.assert_not_called()


@pytest.mark.cpu
def test_invalid_cfg_result_cannot_advance_schedulers():
    request, events = _request("cpu", torch.float32)
    sampler = _sampler()
    sampler.predict_noise_maybe_with_cfg = Mock(return_value=torch.zeros(1))
    with patch.object(sampler_module, "get_classifier_free_guidance_world_size", return_value=2):
        with pytest.raises(RuntimeError, match="must return video and audio"):
            sampler.denoise_step(
                sampler_input=request,
                latent=request.latent,
                audio_latent=request.audio_latent,
                timestep=request.video_t_list[0],
                video_cfg=5.0,
                audio_cfg=3.0,
            )
    assert events == []


def _check_profiler(device, enabled):
    sampler = _sampler()
    request, _ = _request(device, torch.float32)
    pipe = object.__new__(Magi2Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.sampler = sampler
    pipe.setup_diffusion_pipeline_profiler(pipe._PROFILER_TARGETS, enabled)
    with (
        patch.object(sampler_module, "get_classifier_free_guidance_world_size", return_value=1),
        patch.object(profiler_module.logger, "info") as logs,
    ):
        sampler.sample(request)
    if enabled:
        names = [call.args[0] for call in logs.call_args_list]
        assert sum("Magi2Pipeline.sampler.diffuse took" in message for message in names) == 3
        assert sum("Magi2Pipeline.sampler.sample took" in message for message in names) == 1
        # The public dictionary aggregates durations, not individual step samples.
        assert set(pipe.stage_durations) == {"Magi2Pipeline.sampler.diffuse", "Magi2Pipeline.sampler.sample"}
    else:
        logs.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize("enabled", [True, False])
def test_profiler_observes_each_step_only_when_enabled(enabled):
    _check_profiler("cpu", enabled)


@pytest.mark.musa
def test_musa_sampler_parity_and_profiler_boundary():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    for cfg_size in (1, 2):
        for mode in ("plain", "advanced"):
            _check_equivalence("musa", torch.bfloat16, cfg_size, mode)
    _check_profiler("musa", True)
