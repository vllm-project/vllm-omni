# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the LTX-2.3 image-to-video pipeline."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ltx23_request_pipe(cls):
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 99
    return pipe


class TestLTX23ImageToVideoForwardStages:
    def test_forward_resolves_request_image_and_delegates_to_shared_forward_impl(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX23ImageToVideoPipeline)
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

        def fake_forward_impl(req_arg, request_inputs, **kwargs):
            seen["req"] = req_arg
            seen["request_inputs"] = request_inputs
            seen["kwargs"] = kwargs
            return ["i2v-delegated"]

        object.__setattr__(pipe, "_forward_impl", fake_forward_impl)

        output = pipe.forward(req, noise_scale=0.5)

        assert output == ["i2v-delegated"]
        assert seen["req"] is req
        assert seen["request_inputs"].prompt == ["make the image move"]
        assert seen["request_inputs"].negative_prompt == ["jitter"]
        assert seen["kwargs"]["image"] is image
        assert seen["kwargs"]["noise_scale"] == 0.5

    def test_denoise_timestep_kwargs_masks_video_only(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        ts = torch.tensor([2.0, 4.0])
        denoise_ctx = SimpleNamespace(
            conditioning_mask=torch.tensor([[1.0, 0.0]]),
            conditioning_mask_for_model=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        kwargs = pipe._denoise_timestep_kwargs(
            ts,
            SimpleNamespace(cfg_parallel_ready=False),
            denoise_ctx,
        )

        torch.testing.assert_close(kwargs["timestep"], torch.tensor([[0.0, 2.0], [4.0, 0.0]]))
        torch.testing.assert_close(kwargs["audio_timestep"], ts)
        torch.testing.assert_close(kwargs["sigma"], ts)


class TestLTX23ImageToVideoPipeline:
    def test_ltx23_i2v_pipeline_reuses_ltx23_semantics(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        assert issubclass(LTX23ImageToVideoPipeline, LTX23Pipeline)
        assert LTX23ImageToVideoPipeline.support_image_input is True

    def test_ltx23_i2v_rejects_multi_image_prompt_list(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        image = object()

        assert LTX23ImageToVideoPipeline._resolve_single_prompt_image([image]) is image
        with pytest.raises(ValueError, match="exactly one image per prompt"):
            LTX23ImageToVideoPipeline._resolve_single_prompt_image([object(), object()])

    def test_ltx23_i2v_additional_image_resolution_is_tensor_safe(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        image = torch.zeros(1, 3, 4, 4)
        additional = {
            "preprocessed_image": None,
            "pixel_values": image,
            "image": torch.ones_like(image),
        }

        assert LTX23ImageToVideoPipeline._resolve_additional_image(additional) is image

    def test_ltx23_i2v_packed_latents_are_not_noised(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video as ltx23_i2v
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae_spatial_compression_ratio = 1
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1

        def fake_randn_tensor(shape, generator=None, device=None, dtype=None):
            raise AssertionError("packed I2V latents should not be noised")

        monkeypatch.setattr(ltx23_i2v, "randn_tensor", fake_randn_tensor)

        latents = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])

        out, conditioning_mask = pipe.prepare_latents(
            image=None,
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
        torch.testing.assert_close(out, latents)

    def test_ltx23_i2v_5d_latents_noise_preserves_conditioning_frame(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.pipeline_ltx2 as ltx2
        import vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video as ltx23_i2v
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
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

        def fake_randn_tensor(shape, generator=None, device=None, dtype=None):
            return torch.ones(shape, device=device, dtype=dtype)

        monkeypatch.setattr(ltx23_i2v, "randn_tensor", fake_randn_tensor)
        monkeypatch.setattr(ltx2, "randn_tensor", fake_randn_tensor)

        latents = torch.tensor([[[[[10.0]], [[20.0]], [[30.0]]], [[[11.0]], [[21.0]], [[31.0]]]]])

        out, conditioning_mask = pipe.prepare_latents(
            image=None,
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
        torch.testing.assert_close(out, torch.tensor([[[10.0, 11.0], [1.0, 1.0], [1.0, 1.0]]]))

    def test_ltx23_i2v_video_step_preserves_conditioning_frame(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1

        class FakeScheduler:
            def step(self, noise_pred, t, latents, return_dict=False):
                return (latents + noise_pred + t,)

        pipe.scheduler = FakeScheduler()
        latents = torch.tensor([[[1.0], [2.0], [3.0]]])
        noise_pred = torch.full_like(latents, 10.0)

        out = pipe._step_video_latents_i2v(
            noise_pred,
            latents,
            torch.tensor(0.5),
            latent_num_frames=3,
            latent_height=1,
            latent_width=1,
        )

        torch.testing.assert_close(out[:, :1], latents[:, :1])
        torch.testing.assert_close(out[:, 1:], latents[:, 1:] + noise_pred[:, 1:] + 0.5)


class TestLTX23ImageToVideoModule:
    """Test that pipeline_ltx2_3_image2video.py exposes I2V entry points."""

    def test_i2v_classes_importable(self):
        """I2V classes must be importable from the implementation module."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        assert LTX23ImageToVideoPipeline is not None

    def test_post_process_func_importable(self):
        """get_ltx2_post_process_func must be importable for registry lookup."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import get_ltx2_post_process_func

        assert callable(get_ltx2_post_process_func)

    def test_i2v_class_matches_package_export(self):
        """Package export must point at the I2V implementation module."""
        from vllm_omni.diffusion.models.ltx2 import LTX23ImageToVideoPipeline as PackageExported
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import (
            LTX23ImageToVideoPipeline as ModuleExported,
        )

        assert PackageExported is ModuleExported
