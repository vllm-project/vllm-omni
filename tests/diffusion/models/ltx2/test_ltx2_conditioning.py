# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX image-to-video input and conditioning behavior."""

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
    def test_i2v_pil_preprocessing_matches_official_center_crop(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import _preprocess_i2v_pil_images

        pixels = torch.zeros(4, 8, 3, dtype=torch.uint8)
        pixels[:, :, 0] = torch.arange(8, dtype=torch.uint8)
        image = Image.fromarray(pixels.numpy())

        actual = _preprocess_i2v_pil_images(image, height=4, width=4)
        expected = pixels[:, 2:6].permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

        torch.testing.assert_close(actual, expected)

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

    def test_denoise_timestep_kwargs_masks_video_only(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        ts = torch.tensor([2.0, 4.0])
        denoise_ctx = SimpleNamespace(
            conditioning_mask=torch.tensor([[1.0, 0.0]]),
            conditioning_mask_for_model=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        kwargs = pipe._denoise_timestep_kwargs(
            ts,
            SimpleNamespace(cfg_parallel_ready=False),
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

    def test_ltx23_i2v_5d_latents_noise_preserves_conditioning_frame(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as ltx2_conditioning
        import vllm_omni.diffusion.models.ltx2.ltx2_latents as ltx2_latents
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
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
        assert sampled_shapes == [(1, 3, 2)]

    def test_ltx23_i2v_image_noise_is_sampled_after_packing(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_conditioning as ltx2_conditioning
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
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

        assert sampled_shapes == [(1, 3, 2)]
        torch.testing.assert_close(conditioning_mask, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(out, torch.tensor([[[10.0, 11.0], [1.0, 1.0], [1.0, 1.0]]]))

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
