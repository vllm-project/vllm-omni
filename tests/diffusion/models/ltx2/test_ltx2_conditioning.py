# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for LTX image-to-video input and conditioning behavior."""

import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ltx23_request_pipe(cls):
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 99
    pipe.vae_spatial_compression_ratio = 32
    pipe.vae_temporal_compression_ratio = 8
    return pipe


class TestLTXImageToVideoForwardStages:
    def test_legacy_i2v_pil_preprocessing_preserves_aspect_ratio_and_center_crops(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import _preprocess_i2v_pil_images

        pixels = torch.zeros(4, 8, 3, dtype=torch.uint8)
        pixels[:, :, 0] = torch.arange(8, dtype=torch.uint8)
        image = Image.fromarray(pixels.numpy())

        actual = _preprocess_i2v_pil_images(image, height=4, width=4)
        expected = pixels[:, 2:6].permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

        torch.testing.assert_close(actual, expected)

    def test_ltx25_i2v_pil_preprocessing_preserves_aspect_ratio_after_crf(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as conditioning

        pixels = torch.zeros(4, 8, 3, dtype=torch.uint8)
        pixels[:, :, 0] = torch.arange(8, dtype=torch.uint8)
        image = Image.fromarray(pixels.numpy())
        monkeypatch.setattr(conditioning, "_apply_image_conditioning_crf", lambda image_array, _crf: image_array)

        actual = conditioning._preprocess_i2v_pil_images(image, height=4, width=4, crf=18)
        expected = pixels[:, 2:6].permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

        torch.testing.assert_close(actual, expected)

    def test_ltx25_i2v_applies_crf18_before_resize(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as conditioning
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        captured_crfs = []
        captured_prepare_kwargs = []

        def fake_preprocess(image, *, height, width, crf, device, dtype):
            del image, device
            captured_crfs.append(crf)
            return torch.zeros(1, 3, height, width, dtype=dtype)

        def fake_prepare_latents(**kwargs):
            captured_prepare_kwargs.append(kwargs)
            return kwargs["image"], None

        monkeypatch.setattr(conditioning, "_preprocess_i2v_pil_images", fake_preprocess)
        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.model_version = "2.5"
        object.__setattr__(pipe, "transformer", SimpleNamespace(config=SimpleNamespace(in_channels=4)))
        object.__setattr__(pipe, "prepare_latents", fake_prepare_latents)
        request_inputs = SimpleNamespace(
            height=4,
            width=4,
            num_frames=3,
            num_videos_per_prompt=1,
            generator=None,
            latents=None,
        )
        prompt_context = SimpleNamespace(
            batch_size=1,
            positive_connector_prompt_embeds=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
        )

        pipe._prepare_video_latents_stage(
            request_inputs,
            prompt_context,
            device=torch.device("cpu"),
            noise_scale=0.0,
            image=Image.new("RGB", (8, 4)),
        )

        assert captured_crfs == [18]
        assert captured_prepare_kwargs[0]["dtype"] is torch.bfloat16
        assert captured_prepare_kwargs[0]["image"].dtype is torch.bfloat16

    def test_ltx25_i2v_rejects_inputs_that_cannot_receive_crf18(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.model_version = "2.5"

        with pytest.raises(ValueError, match="requires PIL images"):
            pipe._prepare_video_latents_stage(
                SimpleNamespace(),
                SimpleNamespace(),
                device=torch.device("cpu"),
                noise_scale=0.0,
                image=torch.zeros(3, 4, 4),
            )

    def test_ltx25_i2v_tensor_input_can_opt_out_of_crf18(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        captured_prepare_kwargs = []

        def fake_prepare_latents(**kwargs):
            captured_prepare_kwargs.append(kwargs)
            return kwargs["image"], None

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.model_version = "2.5"
        object.__setattr__(pipe, "transformer", SimpleNamespace(config=SimpleNamespace(in_channels=4)))
        object.__setattr__(pipe, "prepare_latents", fake_prepare_latents)
        request_inputs = SimpleNamespace(
            height=4,
            width=4,
            num_frames=3,
            num_videos_per_prompt=1,
            generator=None,
            latents=None,
            image_crf=0,
        )
        prompt_context = SimpleNamespace(
            batch_size=1,
            positive_connector_prompt_embeds=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
        )

        pipe._prepare_video_latents_stage(
            request_inputs,
            prompt_context,
            device=torch.device("cpu"),
            noise_scale=0.0,
            image=torch.zeros(3, 4, 4),
        )

        assert captured_prepare_kwargs[0]["image"].shape == (1, 3, 4, 4)

    def test_ltx25_i2v_crf_preserves_single_pixel_dimension(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import _apply_image_conditioning_crf

        image = torch.zeros(1, 4, 3, dtype=torch.uint8).numpy()
        monkeypatch.setitem(sys.modules, "av", None)

        assert _apply_image_conditioning_crf(image, 18) is image

    def test_ltx25_i2v_reports_missing_pyav(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import _apply_image_conditioning_crf

        monkeypatch.setitem(sys.modules, "av", None)

        with pytest.raises(ImportError, match="PyAV with a libx264 encoder"):
            _apply_image_conditioning_crf(torch.zeros(4, 4, 3, dtype=torch.uint8).numpy(), 18)

    def test_forward_resolves_request_image_and_delegates_to_shared_recipe_runtime(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX2Pipeline)
        image = torch.zeros(3, 8, 8)
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={
                        "prompt": "make the image move",
                        "negative_prompt": "jitter",
                        "multi_modal_data": {"image": image},
                    },
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        num_inference_steps=2,
                    ),
                    request_id="ltx23-i2v-forward-stage-delegation",
                )
            ]
        )
        seen = {}

        def fake_run_recipe(req_arg, request_inputs, **kwargs):
            seen["req"] = req_arg
            seen["request_inputs"] = request_inputs
            seen["kwargs"] = kwargs
            return ["i2v-delegated"]

        object.__setattr__(pipe, "_run_recipe", fake_run_recipe)

        output = pipe.forward(req)

        assert output == ["i2v-delegated"]
        assert seen["req"] is req
        assert seen["request_inputs"].prompt == ["make the image move"]
        assert seen["request_inputs"].negative_prompt == ["jitter"]
        assert seen["kwargs"]["image"] is image
        assert seen["kwargs"]["request_sigmas"] is None

    def test_unified_entry_selects_t2v_without_images_and_rejects_mixed_batches(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        request_inputs = SimpleNamespace(latents=None)
        t2v_req = SimpleNamespace(prompts=["text only", {"prompt": "also text only"}])

        assert pipe._resolve_request_image(t2v_req, None, request_inputs) is None

        mixed_req = SimpleNamespace(
            prompts=[
                "text only",
                {"prompt": "image conditioned", "multi_modal_data": {"image": torch.zeros(3, 8, 8)}},
            ]
        )
        with pytest.raises(ValueError, match="cannot mix text-to-video and image-to-video"):
            pipe._resolve_request_image(mixed_req, None, request_inputs)

    def test_denoise_timestep_kwargs_masks_video_and_keeps_audio_per_token(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        ts = torch.tensor([2.0, 4.0])
        denoise_ctx = SimpleNamespace(
            conditioning_mask=torch.tensor([[1.0, 0.0]]),
            conditioning_mask_for_model=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        kwargs = pipe._denoise_timestep_kwargs(
            ts,
            SimpleNamespace(guidance_parallel_ready=False),
            denoise_ctx,
            video_token_count=2,
            audio_token_count=1,
        )

        torch.testing.assert_close(kwargs["timestep"], torch.tensor([[0.0, 2.0], [4.0, 0.0]]))
        torch.testing.assert_close(kwargs["audio_timestep"], ts[:, None])
        torch.testing.assert_close(kwargs["sigma"], ts)


class TestLTXImageToVideoConditioning:
    def test_ltx23_i2v_supports_image_input(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        assert LTX2Pipeline.support_image_input is True

    def test_ltx25_i2v_encode_uses_diffusion_decoder_statistics(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as ltx2_conditioning
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.use_diffusion_decoder = True
        pipe.vae = SimpleNamespace(
            encode=lambda image: image,
            latents_mean=torch.full((2,), 100.0),
            latents_std=torch.full((2,), 10.0),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        pipe.diffusion_decoder = SimpleNamespace(
            latents_mean=torch.tensor([1.0, 3.0]),
            latents_std=torch.tensor([2.0, 4.0]),
            config=SimpleNamespace(scaling_factor=2.0),
        )
        monkeypatch.setattr(
            ltx2_conditioning,
            "retrieve_latents",
            lambda *_args, **_kwargs: torch.tensor([[[[[3.0]]], [[[7.0]]]]]),
        )

        actual = pipe._encode_i2v_image_latents(
            torch.zeros(1, 3, 1, 1),
            batch_size=1,
            generator=None,
            dtype=torch.float32,
        )

        torch.testing.assert_close(actual, torch.full((1, 2, 1, 1, 1), 2.0))

    def test_ltx23_i2v_rejects_multi_image_prompt_list(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        image = object()

        assert LTX2Pipeline._resolve_single_prompt_image([image]) is image
        with pytest.raises(ValueError, match="exactly one image per prompt"):
            LTX2Pipeline._resolve_single_prompt_image([object(), object()])

    def test_ltx23_i2v_additional_image_resolution_is_tensor_safe(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        image = torch.zeros(1, 3, 4, 4)
        additional = {
            "preprocessed_image": None,
            "pixel_values": image,
            "image": torch.ones_like(image),
        }

        assert LTX2Pipeline._resolve_additional_image(additional) is image

    def test_ltx23_i2v_packed_latents_are_not_noised(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as ltx2_conditioning
        import vllm_omni.diffusion.models.ltx2.ltx2_latents as ltx2_latents
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae_spatial_compression_ratio = 1
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        object.__setattr__(
            pipe,
            "_encode_i2v_image_latents",
            lambda *_args, **_kwargs: torch.tensor([[[[[40.0]]], [[[41.0]]]]]),
        )

        def fake_randn_tensor(shape, generator=None, device=None, dtype=None):
            raise AssertionError("packed I2V latents should not be noised")

        monkeypatch.setattr(ltx2_conditioning, "randn_tensor", fake_randn_tensor)
        monkeypatch.setattr(ltx2_latents, "randn_tensor", fake_randn_tensor)

        latents = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])

        out, conditioning_mask = pipe.prepare_latents(
            image=torch.zeros(1, 3, 1, 1),
            batch_size=1,
            num_channels_latents=2,
            height=1,
            width=1,
            num_frames=3,
            noise_scale=1.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        torch.testing.assert_close(conditioning_mask, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(out, torch.tensor([[[40.0, 41.0], [20.0, 21.0], [30.0, 31.0]]]))

    @pytest.mark.parametrize(
        ("model_version", "sampled_shape"),
        [("2.3", (1, 3, 2)), ("2.5", (1, 3, 2))],
    )
    def test_i2v_5d_latents_noise_uses_packed_rng_layout(self, monkeypatch, model_version, sampled_shape):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as ltx2_conditioning
        import vllm_omni.diffusion.models.ltx2.ltx2_latents as ltx2_latents
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.model_version = model_version
        pipe.vae_spatial_compression_ratio = 1
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        object.__setattr__(
            pipe,
            "_encode_i2v_image_latents",
            lambda *_args, **_kwargs: torch.tensor([[[[[40.0]]], [[[41.0]]]]]),
        )

        sampled_shapes = []

        def fake_randn_tensor(shape, generator=None, device=None, dtype=None):
            sampled_shapes.append(tuple(shape))
            return torch.ones(shape, device=device, dtype=dtype)

        monkeypatch.setattr(ltx2_conditioning, "randn_tensor", fake_randn_tensor)
        monkeypatch.setattr(ltx2_latents, "randn_tensor", fake_randn_tensor)

        latents = torch.tensor([[[[[10.0]], [[20.0]], [[30.0]]], [[[11.0]], [[21.0]], [[31.0]]]]])

        out, conditioning_mask = pipe.prepare_latents(
            image=torch.zeros(1, 3, 1, 1),
            batch_size=1,
            num_channels_latents=2,
            height=1,
            width=1,
            num_frames=3,
            noise_scale=1.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        torch.testing.assert_close(conditioning_mask, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(out, torch.tensor([[[40.0, 41.0], [1.0, 1.0], [1.0, 1.0]]]))
        assert sampled_shapes == [sampled_shape]

    @pytest.mark.parametrize(
        ("model_version", "sampled_shape"),
        [("2.3", (1, 3, 2)), ("2.5", (1, 3, 2))],
    )
    def test_i2v_image_noise_uses_packed_rng_layout(self, monkeypatch, model_version, sampled_shape):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as ltx2_conditioning
        import vllm_omni.diffusion.models.ltx2.ltx2_latents as ltx2_latents
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.model_version = model_version
        pipe.vae_spatial_compression_ratio = 1
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.vae = SimpleNamespace(
            encode=lambda image: image,
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
        )
        monkeypatch.setattr(
            ltx2_conditioning,
            "retrieve_latents",
            lambda *_args, **_kwargs: torch.tensor([[[[[10.0]]], [[[11.0]]]]]),
        )
        sampled_shapes = []

        def fake_randn_tensor(shape, generator=None, device=None, dtype=None):
            sampled_shapes.append(tuple(shape))
            return torch.ones(shape, device=device, dtype=dtype)

        monkeypatch.setattr(ltx2_conditioning, "randn_tensor", fake_randn_tensor)
        monkeypatch.setattr(ltx2_latents, "randn_tensor", fake_randn_tensor)

        out, conditioning_mask = pipe.prepare_latents(
            image=torch.zeros(1, 3, 1, 1),
            batch_size=1,
            num_channels_latents=2,
            height=1,
            width=1,
            num_frames=3,
            noise_scale=1.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert sampled_shapes == [sampled_shape]
        torch.testing.assert_close(conditioning_mask, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(out, torch.tensor([[[10.0, 11.0], [1.0, 1.0], [1.0, 1.0]]]))

    def test_ltx25_re_noise_uses_official_scalar_noise_scale(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_latents as ltx2_latents
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.model_version = "2.5"
        pipe.vae_spatial_compression_ratio = 1
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        object.__setattr__(
            pipe,
            "_encode_i2v_image_latents",
            lambda *_args, **_kwargs: torch.zeros(1, 2, 1, 1, 1, dtype=torch.bfloat16),
        )
        captured = {}

        def fake_create_noised_state(latents, noise_scale, generator=None):
            del generator
            captured["noise_scale"] = noise_scale
            return latents

        monkeypatch.setattr(ltx2_latents, "create_noised_state", fake_create_noised_state)
        pipe.prepare_latents(
            image=torch.zeros(1, 3, 1, 1),
            batch_size=1,
            num_channels_latents=2,
            height=1,
            width=1,
            num_frames=3,
            noise_scale=0.909375,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            latents=torch.zeros(1, 2, 3, 1, 1, dtype=torch.bfloat16),
        )

        assert captured["noise_scale"] == 0.909375

    def test_ltx23_i2v_video_step_preserves_conditioning_frame(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1

        class FakeScheduler:
            sigmas = torch.tensor([1.0, 0.5])

        pipe.scheduler = FakeScheduler()
        latents = torch.tensor([[[1.0], [2.0], [3.0]]])
        noise_pred = torch.full_like(latents, 10.0)

        out = pipe._step_video_latents_i2v(
            noise_pred,
            latents,
            0,
            latent_num_frames=3,
            latent_height=1,
            latent_width=1,
        )

        torch.testing.assert_close(out[:, :1], latents[:, :1])
        torch.testing.assert_close(out[:, 1:], latents[:, 1:] - 0.5 * noise_pred[:, 1:])

    def test_shared_step_adapter_uses_official_euler_for_t2v(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXVideoAudioStepAdapter

        pipeline = SimpleNamespace(scheduler=SimpleNamespace(sigmas=torch.tensor([1.0, 0.5])))
        adapter = LTXVideoAudioStepAdapter(
            pipeline,
            SimpleNamespace(sigmas=torch.tensor([1.0, 0.25])),
            latent_num_frames=1,
            latent_height=1,
            latent_width=1,
            image_conditioned=False,
        )
        video = torch.tensor([[[2.0]]])
        audio = torch.tensor([[[4.0]]])

        ((video, audio),) = adapter.step(
            (torch.tensor([[[2.0]]]), torch.tensor([[[4.0]]])),
            (torch.tensor(1.0), torch.tensor(1.0)),
            (video, audio),
        )

        torch.testing.assert_close(video, torch.tensor([[[1.0]]]))
        torch.testing.assert_close(audio, torch.tensor([[[1.0]]]))

    def test_shared_step_adapter_uses_official_euler_for_i2v_audio(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXVideoAudioStepAdapter

        class FakePipeline:
            scheduler = object()

            def __init__(self):
                self.step_indices = []

            def _step_video_latents_i2v(
                self,
                noise_pred,
                latents,
                step_index,
                latent_num_frames,
                latent_height,
                latent_width,
            ):
                del noise_pred, latent_num_frames, latent_height, latent_width
                self.step_indices.append(step_index)
                return latents

        pipeline = FakePipeline()
        adapter = LTXVideoAudioStepAdapter(
            pipeline,
            SimpleNamespace(sigmas=torch.tensor([1.0, 0.5, 0.0])),
            latent_num_frames=1,
            latent_height=1,
            latent_width=1,
            image_conditioned=True,
        )
        video = torch.tensor([[[1.0]]])
        audio = torch.tensor([[[2.0]]])
        video_velocity = torch.zeros_like(video)
        audio_velocity = torch.tensor([[[4.0]]])

        ((video, audio),) = adapter.step(
            (video_velocity, audio_velocity),
            (torch.tensor(1.0), torch.tensor(1.0)),
            (video, audio),
        )
        torch.testing.assert_close(audio, torch.tensor([[[0.0]]]))

        ((video, audio),) = adapter.step(
            (video_velocity, audio_velocity),
            (torch.tensor(0.5), torch.tensor(0.5)),
            (video, audio),
        )
        torch.testing.assert_close(audio, torch.tensor([[[-2.0]]]))
        assert pipeline.step_indices == [0, 1]

    def test_ltx25_ancestral_adapter_matches_official_seeded_step_and_preserves_i2v_mask(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXVideoAudioStepAdapter

        device = torch.device("cpu")
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        pipeline = SimpleNamespace(device=device, scheduler=SimpleNamespace(sigmas=sigmas))
        mask = torch.tensor([[1.0, 0.0]])
        adapter = LTXVideoAudioStepAdapter(
            pipeline,
            SimpleNamespace(sigmas=sigmas.clone()),
            latent_num_frames=1,
            latent_height=1,
            latent_width=1,
            image_conditioned=True,
            sampler="euler_ancestral",
            generator=torch.Generator(device=device).manual_seed(42),
            conditioning_mask=mask,
        )
        video = torch.tensor([[[3.0], [2.0]]])
        audio = torch.tensor([[[4.0]]])
        video_velocity = torch.tensor([[[7.0], [0.25]]])
        audio_velocity = torch.tensor([[[0.5]]])

        reference_generator = torch.Generator(device=device).manual_seed(10042)
        video_noise = torch.randn(video.shape, generator=reference_generator)
        audio_noise = torch.randn(audio.shape, generator=reference_generator)

        def official_step(sample, velocity, noise):
            sigma = sigmas[0]
            sigma_next = sigmas[1]
            denoised = sample.float() - velocity.float() * sigma
            sigma_down = sigma_next * (sigma_next / sigma)
            ratio = sigma_down / sigma
            deterministic = ratio * sample.float() + (1.0 - ratio) * denoised
            alpha_next = 1.0 - sigma_next
            alpha_down = 1.0 - sigma_down
            coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).sqrt()
            return (alpha_next / alpha_down) * deterministic + noise * coeff

        expected_video = official_step(video, video_velocity, video_noise)
        expected_video[:, :1] = video[:, :1]
        expected_audio = official_step(audio, audio_velocity, audio_noise)

        ((actual_video, actual_audio),) = adapter.step(
            (video_velocity, audio_velocity),
            (sigmas[0], sigmas[0]),
            (video, audio),
        )
        torch.testing.assert_close(actual_video, expected_video)
        torch.testing.assert_close(actual_audio, expected_audio)

        terminal_video_velocity = torch.full_like(actual_video, 0.125)
        terminal_audio_velocity = torch.full_like(actual_audio, 0.25)
        ((terminal_video, terminal_audio),) = adapter.step(
            (terminal_video_velocity, terminal_audio_velocity),
            (sigmas[1], sigmas[1]),
            (actual_video, actual_audio),
        )
        expected_terminal_video = actual_video - terminal_video_velocity * sigmas[1]
        expected_terminal_video[:, :1] = actual_video[:, :1]
        expected_terminal_audio = actual_audio - terminal_audio_velocity * sigmas[1]
        torch.testing.assert_close(terminal_video, expected_terminal_video)
        torch.testing.assert_close(terminal_audio, expected_terminal_audio)

    def test_ltx25_ancestral_quantizes_x0_to_model_dtype_before_fp32_step(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_denoise import _ancestral_euler_step_from_velocity

        sample = torch.tensor([[[3.140625, -1.234375]]], dtype=torch.bfloat16)
        velocity = torch.tensor([[[0.333984375, 2.71875]]], dtype=torch.bfloat16)
        sigmas = torch.tensor([0.73, 0.41, 0.0], dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(10042)
        reference_generator = torch.Generator(device="cpu").manual_seed(10042)

        sigma, sigma_next = sigmas[:2]
        denoised = (sample.float() - velocity.float() * sigma).to(sample.dtype).float()
        sigma_down = sigma_next * (sigma_next / sigma)
        sigma_down_ratio = sigma_down / sigma
        expected = sigma_down_ratio * sample.float() + (1.0 - sigma_down_ratio) * denoised
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        coefficient = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).sqrt()
        noise = torch.randn(
            sample.shape,
            generator=reference_generator,
            dtype=sample.dtype,
            device=sample.device,
        )
        expected = ((alpha_next / alpha_down) * expected + noise.float() * coefficient).to(sample.dtype)

        actual = _ancestral_euler_step_from_velocity(sample, velocity, sigmas, 0, generator)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_i2v_guidance_uses_zero_sigma_for_conditioned_tokens(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXDenoiseContext
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        denoise_ctx = LTXDenoiseContext(
            latents=torch.empty(1, 3, 1),
            audio_latents=torch.empty(1, 1, 1),
            video_coords=torch.empty(1),
            audio_coords=torch.empty(1),
            conditioning_mask=torch.tensor([[1.0, 0.0, 0.0]]),
        )

        actual = pipe._video_guidance_model_sigma(torch.tensor(0.75), denoise_ctx)

        torch.testing.assert_close(actual, torch.tensor([[[0.0], [0.75], [0.75]]]))
