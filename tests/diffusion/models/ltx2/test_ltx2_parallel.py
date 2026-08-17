# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX guidance and forward-parallel behavior."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_velocity_from_x0_uses_official_host_scalar_sigma(mocker):
    from vllm_omni.diffusion.models.ltx2.ltx2_guidance import velocity_from_x0

    sigma = mocker.MagicMock()
    sigma.to.return_value.item.return_value = 0.5

    actual = velocity_from_x0(torch.tensor([1.0]), torch.tensor([0.0]), sigma)

    sigma.to.assert_called_once_with(torch.float32)
    sigma.to.return_value.item.assert_called_once_with()
    torch.testing.assert_close(actual, torch.tensor([2.0]))


class TestCFGParallelHelpers:
    """Test LTX-2.3 CFG helper math without loading model weights."""

    def test_single_guidance_pass_preserves_official_token_layout(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_guidance import _repeat_batch

        tokens = torch.randn(1, 16, 32).transpose(1, 2).contiguous().transpose(1, 2)
        repeated = _repeat_batch(tokens, 1)

        assert repeated is tokens
        assert repeated.stride() == tokens.stride()

    def test_combine_cfg_noise_matches_x0_space_formula(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        video_sample = torch.tensor([[[1.0, -2.0]]])
        audio_sample = torch.tensor([[[0.5, 3.0]]])
        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])
        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)
        scale = 4.0

        video_combined, audio_combined = pipe.combine_cfg_noise(
            (video_pos, audio_pos),
            (video_neg, audio_neg),
            scale,
            video_latents=video_sample,
            audio_latents=audio_sample,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        x0_video_cond = video_sample - video_pos * video_sigma
        x0_video_uncond = video_sample - video_neg * video_sigma
        x0_video_guided = x0_video_cond + (scale - 1) * (x0_video_cond - x0_video_uncond)
        expected_video = (video_sample - x0_video_guided) / video_sigma

        x0_audio_cond = audio_sample - audio_pos * audio_sigma
        x0_audio_uncond = audio_sample - audio_neg * audio_sigma
        x0_audio_guided = x0_audio_cond + (scale - 1) * (x0_audio_cond - x0_audio_uncond)
        expected_audio = (audio_sample - x0_audio_guided) / audio_sigma
        assert torch.allclose(video_combined, expected_video)
        assert torch.allclose(audio_combined, expected_audio)

    @pytest.mark.parametrize("guidance_world_size", [1, 2, 3, 4])
    def test_guidance_parallel_accepts_official_guidance(self, guidance_world_size):
        from vllm_omni.diffusion.models.ltx2.ltx2_guidance import (
            LTX_GUIDANCE_EXECUTOR,
            LTXGuidancePlan,
        )
        from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTX23_ONE_STAGE_RECIPE

        plan = LTXGuidancePlan.build(LTX23_ONE_STAGE_RECIPE.request_guidance)

        LTX_GUIDANCE_EXECUTOR.validate_guidance_world_size(plan, guidance_world_size)

    def test_guidance_parallel_dummy_warms_full_guidance_plan(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2 import ltx2_runtime
        from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTX23_ONE_STAGE_RECIPE
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        monkeypatch.setattr(ltx2_runtime, "get_guidance_parallel_world_size", lambda: 2)
        pipe = object.__new__(LTX2Pipeline)
        req = SimpleNamespace(is_dummy_run=lambda: True)
        request_inputs = SimpleNamespace(guidance=LTX23_ONE_STAGE_RECIPE.request_guidance)

        guidance_parallel_ready = pipe._setup_forward_runtime(req, request_inputs, attention_kwargs=None)

        assert guidance_parallel_ready is True
        assert pipe._guidance_plan.names == ("cond", "uncond", "ptb", "mod")
        assert pipe._guidance_plan.spec.video.cfg_scale == 3.0
        assert pipe._guidance_plan.spec.audio.cfg_scale == 7.0

    def test_guidance_parallel_warns_for_imbalanced_passes(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2 import ltx2_guidance

        spec = ltx2_guidance.LTXGuidanceSpec(
            video=ltx2_guidance.LTXModalityGuidance(
                cfg_scale=3.0,
                stg_scale=1.0,
                stg_blocks=(1,),
            )
        )
        plan = ltx2_guidance.LTXGuidancePlan.build(spec)
        captured = []
        monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_rank", lambda: 0)
        monkeypatch.setattr(ltx2_guidance.logger, "warning_once", lambda *args: captured.append(args))

        ltx2_guidance.LTX_GUIDANCE_EXECUTOR.warn_if_imbalanced(plan, 4, "hq_generate")

        assert plan.names == ("cond", "uncond", "ptb")
        assert len(captured) == 1
        assert captured[0][1:] == ("hq_generate", 3, 4, 1, 75.0)

    @pytest.mark.parametrize("guidance_rank", [0, 1])
    def test_default_guidance_is_sharded_and_combined(self, monkeypatch, guidance_rank):
        from vllm_omni.diffusion.models.ltx2 import ltx2_guidance
        from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXDenoiseContext
        from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
        from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTX23_ONE_STAGE_RECIPE

        executor = ltx2_guidance.LTX_GUIDANCE_EXECUTOR
        plan = ltx2_guidance.LTXGuidancePlan.build(LTX23_ONE_STAGE_RECIPE.request_guidance)
        video_predictions = {
            "cond": torch.tensor([[[0.2, -0.3]]]),
            "uncond": torch.tensor([[[-0.4, 0.1]]]),
            "ptb": torch.tensor([[[0.5, -0.2]]]),
            "mod": torch.tensor([[[-0.1, 0.6]]]),
        }
        audio_predictions = {
            "cond": torch.tensor([[[0.7, -0.2]]]),
            "uncond": torch.tensor([[[0.1, 0.4]]]),
            "ptb": torch.tensor([[[-0.3, 0.2]]]),
            "mod": torch.tensor([[[0.4, -0.5]]]),
        }
        state = LTXAVState(
            video=torch.tensor([[[1.0, -2.0]]]),
            audio=torch.tensor([[[0.5, 3.0]]]),
        )
        denoise_ctx = LTXDenoiseContext(
            latents=state.video,
            audio_latents=state.audio,
            video_coords=torch.zeros(1, 1, 3),
            audio_coords=torch.zeros(1, 1, 1),
        )
        executor.prepare_denoise_context(plan, True, 2, denoise_ctx)

        class FakeGuidanceGroup:
            def __init__(self):
                self.call_index = 0

            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                gathered = (
                    [video_predictions["cond"], video_predictions["uncond"]],
                    [video_predictions["ptb"], video_predictions["mod"]],
                    [audio_predictions["cond"], audio_predictions["uncond"]],
                    [audio_predictions["ptb"], audio_predictions["mod"]],
                )[self.call_index]
                torch.testing.assert_close(tensor, gathered[guidance_rank])
                self.call_index += 1
                return gathered

        group = FakeGuidanceGroup()
        monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_world_size", lambda: 2)
        monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_rank", lambda: guidance_rank)
        monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_group", lambda: group)

        local_names = ("cond", "ptb") if guidance_rank == 0 else ("uncond", "mod")
        captured = {}

        class FakeTransformer:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return (
                    torch.cat([video_predictions[name] for name in local_names]),
                    torch.cat([audio_predictions[name] for name in local_names]),
                )

        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)

        class FakePipeline:
            scheduler = SimpleNamespace(sigmas=torch.stack([video_sigma]))
            transformer = FakeTransformer()

            @staticmethod
            def _transformer_cache_context(name):
                assert name == "guided"
                return nullcontext()

            @staticmethod
            def _build_transformer_kwargs(forward_ctx, denoise_ctx, **kwargs):
                del forward_ctx, denoise_ctx
                return kwargs

            @staticmethod
            def _video_guidance_model_sigma(sigma, denoise_ctx):
                del denoise_ctx
                return sigma

        positive_context = torch.ones(1, 1, 1)
        negative_context = torch.zeros(1, 1, 1)
        forward_ctx = SimpleNamespace(
            prompt_context=SimpleNamespace(
                positive_connector_prompt_embeds=positive_context,
                positive_connector_audio_prompt_embeds=positive_context,
                negative_connector_prompt_embeds=negative_context,
                negative_connector_audio_prompt_embeds=negative_context,
            ),
            attention_kwargs=None,
            audio_scheduler=SimpleNamespace(sigmas=torch.stack([audio_sigma])),
            original_audio_num_frames=state.audio.shape[1],
        )

        actual_video, actual_audio = executor.predict_parallel_guidance(
            FakePipeline(),
            plan,
            index=0,
            timestep=torch.tensor(1.0),
            state=state,
            forward_ctx=forward_ctx,
            denoise_ctx=denoise_ctx,
        )

        expected_video = executor._guide_modality(
            state.video,
            video_predictions,
            video_sigma,
            plan.spec.video,
        )
        expected_audio = executor._guide_modality(
            state.audio,
            audio_predictions,
            audio_sigma,
            plan.spec.audio,
        )
        torch.testing.assert_close(actual_video, expected_video)
        torch.testing.assert_close(actual_audio, expected_audio)
        assert group.call_index == 4
        assert captured["hidden_states"].shape[0] == 2
        perturbations = captured["attention_kwargs"]["ltx_perturbation_kwargs"]
        if guidance_rank == 0:
            assert "video_self_attention_mask" in perturbations
            assert "a2v_cross_attention_mask" not in perturbations
        else:
            assert "video_self_attention_mask" not in perturbations
            assert "a2v_cross_attention_mask" in perturbations


class TestCFGParallelForwardPath:
    """Test the LTX-2.3 CFG-parallel denoising path without loading model weights."""

    def test_forward_collates_request_prompt_embeds_and_mask_aliases(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2 import ltx2_runtime
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2 as ltx23
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = object.__new__(ltx23.LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.device = torch.device("cpu")
        pipe.tokenizer_max_length = 4
        pipe.vae_spatial_compression_ratio = 32
        pipe.vae_temporal_compression_ratio = 8
        monkeypatch.setattr(ltx2_runtime, "get_guidance_parallel_world_size", lambda: 1)

        class StopAtEncodePromptError(Exception):
            pass

        captured = {}

        def fake_encode_prompt(**kwargs):
            captured.update(kwargs)
            raise StopAtEncodePromptError

        object.__setattr__(pipe, "encode_prompt", fake_encode_prompt)

        prompt_embeds_a = torch.zeros(2, 3)
        prompt_embeds_b = torch.ones(2, 3)
        negative_prompt_embeds_a = torch.full((2, 3), 2.0)
        negative_prompt_embeds_b = torch.full((2, 3), 3.0)
        prompt_attention_mask_a = torch.tensor([True, True])
        prompt_attention_mask_b = torch.tensor([True, False])
        negative_attention_mask_a = torch.tensor([False, True])
        negative_attention_mask_b = torch.tensor([False, False])

        requests = [
            OmniDiffusionRequest(
                prompt={
                    "prompt": "prompt-a",
                    "negative_prompt": "negative-a",
                    "prompt_embeds": prompt_embeds_a,
                    "negative_prompt_embeds": negative_prompt_embeds_a,
                    "prompt_attention_mask": prompt_attention_mask_a,
                    "negative_prompt_attention_mask": negative_attention_mask_a,
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=32,
                    width=32,
                    num_frames=1,
                    frame_rate=1.0,
                    num_inference_steps=2,
                ),
                request_id="ltx23-prompt-local-a",
            ),
            OmniDiffusionRequest(
                prompt={
                    "prompt": "prompt-b",
                    "negative_prompt": "negative-b",
                    "prompt_embeds": prompt_embeds_b,
                    "negative_prompt_embeds": negative_prompt_embeds_b,
                    "attention_mask": prompt_attention_mask_b,
                    "negative_attention_mask": negative_attention_mask_b,
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=32,
                    width=32,
                    num_frames=1,
                    frame_rate=1.0,
                    num_inference_steps=2,
                ),
                request_id="ltx23-prompt-local-b",
            ),
        ]

        with pytest.raises(StopAtEncodePromptError):
            pipe.forward(DiffusionRequestBatch(requests=requests))

        assert captured["prompt"] is None
        assert captured["negative_prompt"] is None
        torch.testing.assert_close(
            captured["prompt_embeds"],
            torch.stack([prompt_embeds_a, prompt_embeds_b], dim=0),
        )
        torch.testing.assert_close(
            captured["negative_prompt_embeds"],
            torch.stack([negative_prompt_embeds_a, negative_prompt_embeds_b], dim=0),
        )
        torch.testing.assert_close(
            captured["prompt_attention_mask"],
            torch.stack([prompt_attention_mask_a, prompt_attention_mask_b], dim=0),
        )
        torch.testing.assert_close(
            captured["negative_prompt_attention_mask"],
            torch.stack([negative_attention_mask_a, negative_attention_mask_b], dim=0),
        )

    @pytest.mark.parametrize(("cfg_rank", "expected_prompt_value"), [(0, 1.0), (1, 0.0)])
    @pytest.mark.parametrize(
        ("frame_rate_input", "audio_sampling_rate", "expected_frame_rate"),
        [(1.0, 1, 1.0), (None, 24, 24.0)],
    )
    def test_forward_cfg_parallel_uses_shared_video_audio_euler_step(
        self,
        monkeypatch,
        cfg_rank,
        expected_prompt_value,
        frame_rate_input,
        audio_sampling_rate,
        expected_frame_rate,
    ):
        from vllm_omni.diffusion.models.ltx2 import ltx2_denoise, ltx2_guidance, ltx2_runtime
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2 as ltx23
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = object.__new__(ltx23.LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.device = torch.device("cpu")
        pipe.tokenizer_max_length = 1
        pipe.vae_spatial_compression_ratio = 32
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.audio_sampling_rate = audio_sampling_rate
        pipe.audio_hop_length = 1
        pipe.audio_vae_temporal_compression_ratio = 1
        pipe.audio_vae_mel_compression_ratio = 1
        pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
        pipe.tokenizer = SimpleNamespace(padding_side="left")
        pipe.vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        pipe.audio_vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(mel_bins=2, latent_channels=1),
        )

        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])

        class FakeCfgGroup:
            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                if torch.equal(tensor, video_pos) or torch.equal(tensor, video_neg):
                    return [video_pos, video_neg]
                if torch.equal(tensor, audio_pos) or torch.equal(tensor, audio_neg):
                    return [audio_pos, audio_neg]
                raise AssertionError(f"Unexpected gathered tensor: {tensor}")

        monkeypatch.setattr(ltx2_runtime, "get_guidance_parallel_world_size", lambda: 2)
        monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_world_size", lambda: 2)
        monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_rank", lambda: cfg_rank)
        monkeypatch.setattr(ltx2_guidance, "get_guidance_parallel_group", lambda: FakeCfgGroup())

        def fake_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas=None, mu=None):
            scheduler.sigmas = torch.tensor([0.25, 0.25], device=device)
            return torch.tensor([1.0, 0.5], device=device), 2

        monkeypatch.setattr(ltx2_denoise, "retrieve_timesteps", fake_retrieve_timesteps)

        class FakeScheduler:
            def __init__(self, name="video", calls=None):
                self.name = name
                self.calls = [] if calls is None else calls
                self.config = {
                    "max_image_seq_len": 4096,
                    "base_image_seq_len": 1024,
                    "base_shift": 0.95,
                    "max_shift": 2.05,
                }
                self.sigmas = torch.tensor([0.25, 0.25])

            def __deepcopy__(self, memo):
                return FakeScheduler("audio", self.calls)

            def step(self, noise_pred, t, latents, return_dict=False, generator=None):
                self.calls.append((self.name, noise_pred.clone(), t.clone(), latents.clone()))
                return (latents - noise_pred,)

        class FakeConnectors:
            def to(self, device):
                return self

            def __call__(self, prompt_embeds, prompt_attention_mask, padding_side):
                assert padding_side == "left"
                assert prompt_embeds.shape[0] == 1
                return prompt_embeds, prompt_embeds, prompt_attention_mask

        rope_video_fps: list[float] = []

        class FakeRope:
            def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps):
                rope_video_fps.append(fps)
                return torch.zeros(batch_size, num_frames * height * width, 3, device=device)

            def prepare_audio_coords(self, batch_size, num_frames, device):
                return torch.zeros(batch_size, num_frames, 1, device=device)

        class FakeTransformer:
            def __init__(self):
                self.config = SimpleNamespace(in_channels=2)
                self.rope = FakeRope()
                self.audio_rope = FakeRope()
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                expected_prompt = torch.full((1, 1, 1), expected_prompt_value)
                torch.testing.assert_close(kwargs["encoder_hidden_states"], expected_prompt)
                torch.testing.assert_close(kwargs["audio_encoder_hidden_states"], expected_prompt)
                assert kwargs["hidden_states"].shape == (1, 1, 2)
                assert kwargs["audio_hidden_states"].shape == (1, 1, 2)
                if cfg_rank == 0:
                    return video_pos, audio_pos
                return video_neg, audio_neg

        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self):
                pass

        pipe.scheduler = FakeScheduler()
        pipe.connectors = FakeConnectors()
        pipe.transformer = FakeTransformer()
        object.__setattr__(pipe, "progress_bar", lambda total: DummyProgress())

        def fake_encode_prompt(**kwargs):
            return (
                torch.ones(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
            )

        object.__setattr__(pipe, "encode_prompt", fake_encode_prompt)

        video_latents = torch.tensor([[[1.0, -2.0]]])
        audio_latents = torch.tensor([[[0.5, 3.0]]])
        req = OmniDiffusionRequest(
            prompt={"prompt": "prompt", "negative_prompt": "negative"},
            sampling_params=OmniDiffusionSamplingParams(
                height=32,
                width=32,
                num_frames=1,
                frame_rate=frame_rate_input,
                num_inference_steps=2,
                extra_args={
                    "video_cfg_scale": 3.0,
                    "audio_cfg_scale": 7.0,
                    "video_stg_scale": 0.0,
                    "audio_stg_scale": 0.0,
                    "video_modality_scale": 1.0,
                    "audio_modality_scale": 1.0,
                    "video_rescale_scale": 0.0,
                    "audio_rescale_scale": 0.0,
                },
                latents=video_latents,
                audio_latents=audio_latents,
                output_type="latent",
            ),
            request_id="ltx23-cfg-parallel-forward-test",
        )

        output = pipe.forward(DiffusionRequestBatch(requests=[req]), sigmas=[0.25, 0.25])[0]

        expected_video_noise = ltx2_guidance.combine_velocity_via_x0(
            video_latents,
            video_pos,
            video_neg,
            pipe.scheduler.sigmas[0],
            3.0,
        )
        expected_audio_noise = ltx2_guidance.combine_velocity_via_x0(
            audio_latents,
            audio_pos,
            audio_neg,
            pipe.scheduler.sigmas[0],
            7.0,
        )
        assert pipe.scheduler.calls == []
        assert len(pipe.transformer.calls) == 2

        video_out, audio_out = output.output
        sigma_delta = pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[0]
        expected_video = video_latents + expected_video_noise * sigma_delta
        expected_audio = audio_latents + expected_audio_noise * sigma_delta
        torch.testing.assert_close(video_out, expected_video.reshape(1, 2, 1, 1, 1))
        torch.testing.assert_close(audio_out, expected_audio.reshape(1, 1, 1, 2))

        # fps regression guard: an omitted request fps (frame_rate_input=None) must resolve
        # to the model's own 24.0 default, not crash on None; a provided rate is passed through.
        assert rope_video_fps
        assert all(fps == expected_frame_rate for fps in rope_video_fps)
