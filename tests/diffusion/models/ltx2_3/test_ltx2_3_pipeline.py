# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX-2.3 pipeline integration.

These tests verify:
- Pipeline is properly registered in the diffusion registry
- Post-process function is registered
- Cache-DiT enablers are registered
- Pipeline does NOT inherit from LTX2Pipeline
- Vocoder sample rate detection logic
- Re-export module works correctly
"""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ltx23_pipeline(sequence_parallel_size: int = 1):
    from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline

    pipeline = object.__new__(LTX23Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_spatial_compression_ratio = 32
    pipeline.vae_temporal_compression_ratio = 8
    pipeline.transformer_spatial_patch_size = 1
    pipeline.transformer_temporal_patch_size = 1
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=sequence_parallel_size))
    pipeline.audio_vae = SimpleNamespace(
        latents_mean=torch.tensor(0.0),
        latents_std=torch.tensor(1.0),
    )
    return pipeline


def _make_ltx23_request_pipe(cls):
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 99
    return pipe


def _resolve_request_inputs_for_test(pipe, req, guidance_scale=None):
    from vllm_omni.diffusion.models.ltx2_3.ltx2_3_recipes import LTX23_ONE_STAGE_RECIPE

    return pipe._resolve_request_inputs(
        req,
        prompt=None,
        negative_prompt=None,
        height=None,
        width=None,
        num_frames=None,
        frame_rate=None,
        num_inference_steps=None,
        timesteps=None,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=LTX23_ONE_STAGE_RECIPE.decode_timestep,
        decode_noise_scale=LTX23_ONE_STAGE_RECIPE.decode_noise_scale,
        output_type="np",
        max_sequence_length=None,
    )


class TestLTX23RequestParsing:
    def test_official_ltx23_recipes_capture_stage_defaults(self):
        from vllm_omni.diffusion.models.ltx2_3.ltx2_3_recipes import (
            LTX23_HQ_TWO_STAGE_RECIPE,
            LTX23_ONE_STAGE_RECIPE,
            LTX23_STAGE_2_DISTILLED_SIGMAS,
            LTX23_TWO_STAGE_RECIPE,
        )

        assert LTX23_ONE_STAGE_RECIPE.name == "one_stage_official"
        assert (LTX23_ONE_STAGE_RECIPE.height, LTX23_ONE_STAGE_RECIPE.width) == (512, 768)
        assert LTX23_ONE_STAGE_RECIPE.num_inference_steps == 30
        assert LTX23_ONE_STAGE_RECIPE.video_guidance.stg_blocks == (28,)
        assert len(LTX23_ONE_STAGE_RECIPE.stages) == 1

        assert LTX23_TWO_STAGE_RECIPE.name == "two_stage_official"
        assert (LTX23_TWO_STAGE_RECIPE.height, LTX23_TWO_STAGE_RECIPE.width) == (1024, 1536)
        assert LTX23_TWO_STAGE_RECIPE.stages[0].width_scale == 0.5
        assert LTX23_TWO_STAGE_RECIPE.stages[1].sigma_values == LTX23_STAGE_2_DISTILLED_SIGMAS
        assert LTX23_TWO_STAGE_RECIPE.stages[1].uses_spatial_upsampler

        assert LTX23_HQ_TWO_STAGE_RECIPE.name == "two_stage_hq_official"
        assert (LTX23_HQ_TWO_STAGE_RECIPE.height, LTX23_HQ_TWO_STAGE_RECIPE.width) == (1088, 1920)
        assert LTX23_HQ_TWO_STAGE_RECIPE.num_inference_steps == 15
        assert LTX23_HQ_TWO_STAGE_RECIPE.video_guidance.stg_scale == 0.0
        assert LTX23_HQ_TWO_STAGE_RECIPE.audio_guidance.rescale_scale == 1.0
        assert LTX23_HQ_TWO_STAGE_RECIPE.stages[0].sampler == "res2s"
        assert LTX23_HQ_TWO_STAGE_RECIPE.stages[0].distilled_lora_strength == 0.25
        assert LTX23_HQ_TWO_STAGE_RECIPE.stages[1].distilled_lora_strength == 0.5

    def test_request_input_resolution_uses_official_ltx23_defaults(self):
        from vllm_omni.diffusion.models.ltx2_3.ltx2_3_recipes import LTX23_ONE_STAGE_RECIPE
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={"prompt": "prompt", "negative_prompt": "negative"},
                    sampling_params=OmniDiffusionSamplingParams(num_frames=None),
                    request_id="ltx23-defaults",
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(_make_ltx23_request_pipe(LTX23Pipeline), req)

        assert resolved.height == LTX23_ONE_STAGE_RECIPE.height
        assert resolved.width == LTX23_ONE_STAGE_RECIPE.width
        assert resolved.num_frames == LTX23_ONE_STAGE_RECIPE.num_frames
        assert resolved.frame_rate == LTX23_ONE_STAGE_RECIPE.frame_rate
        assert resolved.num_inference_steps == LTX23_ONE_STAGE_RECIPE.num_inference_steps
        assert resolved.guidance_scale == LTX23_ONE_STAGE_RECIPE.video_guidance.cfg_scale
        assert resolved.guidance_params.video_cfg_scale == LTX23_ONE_STAGE_RECIPE.video_guidance.cfg_scale
        assert resolved.guidance_params.audio_cfg_scale == LTX23_ONE_STAGE_RECIPE.audio_guidance.cfg_scale
        assert resolved.guidance_params.video_stg_scale == LTX23_ONE_STAGE_RECIPE.video_guidance.stg_scale
        assert resolved.guidance_params.audio_stg_scale == LTX23_ONE_STAGE_RECIPE.audio_guidance.stg_scale
        assert resolved.guidance_params.video_modality_scale == LTX23_ONE_STAGE_RECIPE.video_guidance.modality_scale
        assert resolved.guidance_params.audio_modality_scale == LTX23_ONE_STAGE_RECIPE.audio_guidance.modality_scale
        assert resolved.guidance_params.video_rescale_scale == LTX23_ONE_STAGE_RECIPE.video_guidance.rescale_scale
        assert resolved.guidance_params.audio_rescale_scale == LTX23_ONE_STAGE_RECIPE.audio_guidance.rescale_scale
        assert resolved.guidance_params.video_stg_blocks == LTX23_ONE_STAGE_RECIPE.video_guidance.stg_blocks
        assert resolved.guidance_params.audio_stg_blocks == LTX23_ONE_STAGE_RECIPE.audio_guidance.stg_blocks
        assert resolved.decode_timestep == LTX23_ONE_STAGE_RECIPE.decode_timestep
        assert resolved.decode_noise_scale == LTX23_ONE_STAGE_RECIPE.decode_noise_scale

    def test_dummy_request_input_resolution_uses_pipeline_recipe_guidance_defaults(self):
        from dataclasses import replace

        from vllm_omni.diffusion.models.ltx2_3.ltx2_3_recipes import LTX23_ONE_STAGE_RECIPE
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        pipe._ltx23_recipe = replace(
            LTX23_ONE_STAGE_RECIPE,
            video_guidance=replace(LTX23_ONE_STAGE_RECIPE.video_guidance, cfg_scale=5.0),
            audio_guidance=replace(LTX23_ONE_STAGE_RECIPE.audio_guidance, cfg_scale=9.0),
        )
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="dummy run",
                    sampling_params=OmniDiffusionSamplingParams(guidance_scale=0.0),
                    request_id=DUMMY_DIFFUSION_REQUEST_ID,
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(pipe, req)

        assert resolved.guidance_scale == 5.0
        assert resolved.guidance_params.video_cfg_scale == 5.0
        assert resolved.guidance_params.audio_cfg_scale == 9.0

    def test_request_input_resolution_uses_split_audio_video_recipe_guidance_defaults(self):
        from vllm_omni.diffusion.models.ltx2_3.ltx2_3_recipes import LTX23_HQ_TWO_STAGE_RECIPE
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        pipe._ltx23_recipe = LTX23_HQ_TWO_STAGE_RECIPE
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={"prompt": "prompt", "negative_prompt": "negative"},
                    sampling_params=OmniDiffusionSamplingParams(),
                    request_id="ltx23-hq-guidance-defaults",
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(pipe, req)

        assert resolved.guidance_params.video_rescale_scale == LTX23_HQ_TWO_STAGE_RECIPE.video_guidance.rescale_scale
        assert resolved.guidance_params.audio_rescale_scale == LTX23_HQ_TWO_STAGE_RECIPE.audio_guidance.rescale_scale
        assert resolved.guidance_params.video_stg_blocks == LTX23_HQ_TWO_STAGE_RECIPE.video_guidance.stg_blocks
        assert resolved.guidance_params.audio_stg_blocks == LTX23_HQ_TWO_STAGE_RECIPE.audio_guidance.stg_blocks

    def test_request_input_resolution_allows_ltx23_guidance_overrides(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={"prompt": "prompt", "negative_prompt": "negative"},
                    sampling_params=OmniDiffusionSamplingParams(
                        guidance_scale=4.0,
                        extra_args={
                            "video_cfg_scale": 4.0,
                            "audio_cfg_scale": 4.0,
                            "video_stg_scale": 0.0,
                            "audio_stg_scale": 0.0,
                            "video_modality_scale": 1.0,
                            "audio_modality_scale": 1.0,
                            "video_rescale_scale": 0.0,
                            "audio_rescale_scale": 0.0,
                            "video_stg_blocks": [],
                            "audio_stg_blocks": [],
                        },
                    ),
                    request_id="ltx23-overrides",
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(_make_ltx23_request_pipe(LTX23Pipeline), req)

        assert resolved.guidance_scale == 4.0
        assert resolved.guidance_params.video_cfg_scale == 4.0
        assert resolved.guidance_params.audio_cfg_scale == 4.0
        assert resolved.guidance_params.video_stg_scale == 0.0
        assert resolved.guidance_params.audio_stg_scale == 0.0
        assert resolved.guidance_params.video_modality_scale == 1.0
        assert resolved.guidance_params.audio_modality_scale == 1.0
        assert resolved.guidance_params.video_rescale_scale == 0.0
        assert resolved.guidance_params.audio_rescale_scale == 0.0
        assert resolved.guidance_params.video_stg_blocks == ()
        assert resolved.guidance_params.audio_stg_blocks == ()

    def test_request_input_resolution_explicit_guidance_scale_keeps_legacy_cfg_fallback(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={"prompt": "prompt", "negative_prompt": "negative"},
                    sampling_params=OmniDiffusionSamplingParams(guidance_scale=4.0),
                    request_id="ltx23-guidance-scale-fallback",
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(_make_ltx23_request_pipe(LTX23Pipeline), req)

        assert resolved.guidance_params.video_cfg_scale == 4.0
        assert resolved.guidance_params.audio_cfg_scale == 4.0

    def test_request_input_resolution_rejects_mixed_explicit_guidance_scale_batch(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={"prompt": "prompt 0", "negative_prompt": "negative"},
                    sampling_params=OmniDiffusionSamplingParams(guidance_scale=4.0),
                    request_id="ltx23-mixed-guidance-scale-0",
                ),
                OmniDiffusionRequest(
                    prompt={"prompt": "prompt 1", "negative_prompt": "negative"},
                    sampling_params=OmniDiffusionSamplingParams(guidance_scale=5.0),
                    request_id="ltx23-mixed-guidance-scale-1",
                ),
            ]
        )

        with pytest.raises(ValueError, match="homogeneous guidance"):
            _resolve_request_inputs_for_test(pipe, req)

    def test_t2v_and_i2v_share_request_input_resolution(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        prompt_embeds = torch.tensor([[1.0, 2.0]])
        negative_prompt_embeds = torch.tensor([[3.0, 4.0]])
        prompt_attention_mask = torch.tensor([True, False])
        negative_attention_mask = torch.tensor([False, True])
        video_latents = torch.ones(1, 2, 3)
        audio_latents = torch.zeros(1, 2, 3)
        generator = torch.Generator().manual_seed(123)

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={
                        "prompt": "shared prompt",
                        "negative_prompt": "shared negative",
                        "additional_information": {
                            "prompt_embeds": prompt_embeds,
                            "negative_prompt_embeds": negative_prompt_embeds,
                            "attention_mask": prompt_attention_mask,
                            "negative_attention_mask": negative_attention_mask,
                        },
                    },
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        frame_rate=12.5,
                        num_inference_steps=1,
                        num_outputs_per_prompt=2,
                        guidance_scale=6.0,
                        generator=generator,
                        latents=video_latents,
                        extra_args={"audio_latents": audio_latents},
                        decode_timestep=[0.1],
                        decode_noise_scale=[0.2],
                        output_type="latent",
                        max_sequence_length=17,
                    ),
                    request_id="ltx23-shared-request-inputs",
                )
            ]
        )

        resolved_t2v = _resolve_request_inputs_for_test(
            _make_ltx23_request_pipe(LTX23Pipeline),
            req,
        )
        resolved_i2v = _resolve_request_inputs_for_test(
            _make_ltx23_request_pipe(LTX23ImageToVideoPipeline),
            req,
        )

        assert resolved_i2v.prompt == resolved_t2v.prompt is None
        assert resolved_i2v.negative_prompt == resolved_t2v.negative_prompt is None
        assert resolved_i2v.height == resolved_t2v.height == 384
        assert resolved_i2v.width == resolved_t2v.width == 512
        assert resolved_i2v.num_frames == resolved_t2v.num_frames == 25
        assert resolved_i2v.frame_rate == resolved_t2v.frame_rate == 12.5
        assert resolved_i2v.num_inference_steps == resolved_t2v.num_inference_steps == 2
        assert resolved_i2v.guidance_scale == resolved_t2v.guidance_scale == 6.0
        assert resolved_i2v.num_videos_per_prompt == resolved_t2v.num_videos_per_prompt == 2
        assert isinstance(resolved_i2v.generator, list)
        assert isinstance(resolved_t2v.generator, list)
        assert [gen.initial_seed() for gen in resolved_i2v.generator] == [123, 123]
        assert [gen.initial_seed() for gen in resolved_t2v.generator] == [123, 123]
        assert resolved_i2v.decode_timestep == resolved_t2v.decode_timestep == [0.1]
        assert resolved_i2v.decode_noise_scale == resolved_t2v.decode_noise_scale == [0.2]
        assert resolved_i2v.output_type == resolved_t2v.output_type == "latent"
        assert resolved_i2v.max_sequence_length == resolved_t2v.max_sequence_length == 17
        torch.testing.assert_close(resolved_i2v.latents, video_latents)
        torch.testing.assert_close(resolved_t2v.latents, video_latents)
        torch.testing.assert_close(resolved_i2v.audio_latents, audio_latents)
        torch.testing.assert_close(resolved_t2v.audio_latents, audio_latents)
        torch.testing.assert_close(resolved_i2v.prompt_embeds, torch.stack([prompt_embeds]))
        torch.testing.assert_close(resolved_t2v.prompt_embeds, torch.stack([prompt_embeds]))
        torch.testing.assert_close(resolved_i2v.negative_prompt_embeds, torch.stack([negative_prompt_embeds]))
        torch.testing.assert_close(resolved_t2v.negative_prompt_embeds, torch.stack([negative_prompt_embeds]))
        torch.testing.assert_close(resolved_i2v.prompt_attention_mask, torch.stack([prompt_attention_mask]))
        torch.testing.assert_close(resolved_t2v.prompt_attention_mask, torch.stack([prompt_attention_mask]))
        torch.testing.assert_close(
            resolved_i2v.negative_prompt_attention_mask,
            torch.stack([negative_attention_mask]),
        )
        torch.testing.assert_close(
            resolved_t2v.negative_prompt_attention_mask,
            torch.stack([negative_attention_mask]),
        )

    def test_request_input_resolution_rejects_mixed_precomputed_prompt_fields(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={
                        "prompt": "with embeds",
                        "additional_information": {"prompt_embeds": torch.tensor([[1.0]])},
                    },
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        num_inference_steps=2,
                    ),
                    request_id="ltx23-mixed-precomputed-fields-0",
                ),
                OmniDiffusionRequest(
                    prompt={"prompt": "without embeds"},
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        num_inference_steps=2,
                    ),
                    request_id="ltx23-mixed-precomputed-fields-1",
                ),
            ]
        )

        with pytest.raises(ValueError, match="mix of provided and missing prompt_embeds"):
            _resolve_request_inputs_for_test(pipe, req)


class TestLTX23ForwardStages:
    def test_encode_prompt_batches_positive_and_negative_prompts_for_cfg(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        calls = []

        def fake_get_gemma_prompt_embeds(
            *,
            prompt,
            num_videos_per_prompt,
            max_sequence_length,
            device,
            dtype=None,
        ):
            calls.append(
                {
                    "prompt": prompt,
                    "num_videos_per_prompt": num_videos_per_prompt,
                    "max_sequence_length": max_sequence_length,
                    "device": device,
                    "dtype": dtype,
                }
            )
            values = torch.arange(len(prompt), dtype=torch.float32, device=device).view(-1, 1, 1)
            mask = torch.ones((len(prompt), 1), dtype=torch.bool, device=device)
            return values, mask

        pipe._get_gemma_prompt_embeds = fake_get_gemma_prompt_embeds

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=["p0", "p1"],
            negative_prompt=["n0", "n1"],
            do_classifier_free_guidance=True,
            num_videos_per_prompt=2,
            max_sequence_length=1024,
            device=torch.device("cpu"),
        )

        assert len(calls) == 1
        assert calls[0]["prompt"] == ["p0", "p1", "n0", "n1"]
        assert calls[0]["num_videos_per_prompt"] == 1
        torch.testing.assert_close(prompt_embeds[:, 0, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        torch.testing.assert_close(negative_prompt_embeds[:, 0, 0], torch.tensor([2.0, 2.0, 3.0, 3.0]))
        torch.testing.assert_close(
            prompt_attention_mask,
            torch.ones((4, 1), dtype=torch.bool),
        )
        torch.testing.assert_close(
            negative_prompt_attention_mask,
            torch.ones((4, 1), dtype=torch.bool),
        )

    def test_encode_prompt_repeats_precomputed_embeds_for_num_outputs(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        prompt_embeds = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
        negative_prompt_embeds = torch.tensor([[[-1.0], [-2.0]], [[-3.0], [-4.0]]])
        prompt_attention_mask = torch.tensor([[True, False], [False, True]])
        negative_prompt_attention_mask = torch.tensor([[False, False], [True, True]])

        (
            repeated_prompt_embeds,
            repeated_prompt_attention_mask,
            repeated_negative_prompt_embeds,
            repeated_negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=None,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            repeated_prompt_embeds,
            torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]], [[3.0], [4.0]], [[3.0], [4.0]]]),
        )
        torch.testing.assert_close(
            repeated_negative_prompt_embeds,
            torch.tensor([[[-1.0], [-2.0]], [[-1.0], [-2.0]], [[-3.0], [-4.0]], [[-3.0], [-4.0]]]),
        )
        torch.testing.assert_close(
            repeated_prompt_attention_mask,
            torch.tensor([[True, False], [True, False], [False, True], [False, True]]),
        )
        torch.testing.assert_close(
            repeated_negative_prompt_attention_mask,
            torch.tensor([[False, False], [False, False], [True, True], [True, True]]),
        )

    def test_t2v_forward_delegates_to_shared_forward_impl(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="stage refactor prompt",
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        frame_rate=16,
                        num_inference_steps=3,
                        guidance_scale=4.5,
                        output_type="latent",
                    ),
                    request_id="ltx23-t2v-forward-stage-delegation",
                )
            ]
        )
        sigmas = [1.0, 0.5]
        timesteps = [1000, 500]
        attention_kwargs = {"scale": 1.0}
        seen = {}

        def fake_forward_impl(req_arg, request_inputs, **kwargs):
            seen["req"] = req_arg
            seen["request_inputs"] = request_inputs
            seen["kwargs"] = kwargs
            return ["delegated"]

        object.__setattr__(pipe, "_forward_impl", fake_forward_impl)

        output = pipe.forward(
            req,
            sigmas=sigmas,
            timesteps=timesteps,
            noise_scale=0.25,
            attention_kwargs=attention_kwargs,
        )

        assert output == ["delegated"]
        assert seen["req"] is req
        assert seen["request_inputs"].height == 384
        assert seen["request_inputs"].width == 512
        assert seen["request_inputs"].num_frames == 25
        assert seen["request_inputs"].frame_rate == 16.0
        assert seen["request_inputs"].guidance_scale == 4.5
        assert seen["request_inputs"].output_type == "latent"
        assert seen["kwargs"] == {
            "noise_scale": 0.25,
            "sigmas": sigmas,
            "timesteps": timesteps,
            "attention_kwargs": attention_kwargs,
        }


class TestPipelineIndependence:
    """Verify LTX23Pipeline is fully independent from LTX2Pipeline."""

    def test_ltx23_pipeline_does_not_inherit_from_ltx2(self):
        """LTX23Pipeline must NOT inherit from LTX2Pipeline."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline

        assert not issubclass(LTX23Pipeline, LTX2Pipeline), (
            "LTX23Pipeline should be fully independent and not inherit from LTX2Pipeline"
        )

    def test_ltx23_pipeline_is_nn_module(self):
        """LTX23Pipeline must be an nn.Module."""
        import torch.nn as nn

        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline

        assert issubclass(LTX23Pipeline, nn.Module)

    def test_ltx23_pipeline_has_progress_bar(self):
        """LTX23Pipeline must mix in ProgressBarMixin."""
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

        assert issubclass(LTX23Pipeline, ProgressBarMixin)

    def test_ltx23_pipeline_declares_offload_components(self):
        """LTX23Pipeline must expose LTX-2.3-specific modules to offload discovery."""
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

        pipe = object.__new__(LTX23Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer = torch.nn.Linear(1, 1)
        pipe.text_encoder = torch.nn.Linear(1, 1)
        pipe.connectors = torch.nn.Linear(1, 1)
        pipe.vae = torch.nn.Linear(1, 1)
        pipe.audio_vae = torch.nn.Linear(1, 1)
        pipe.vocoder = torch.nn.Linear(1, 1)

        modules = ModuleDiscovery.discover(pipe)

        assert LTX23Pipeline._dit_modules == ["transformer"]
        assert LTX23Pipeline._encoder_modules == ["text_encoder", "connectors"]
        assert LTX23Pipeline._vae_modules == ["vae", "audio_vae"]
        assert LTX23Pipeline._resident_modules == ["vocoder"]
        assert modules.dit_names == ["transformer"]
        assert modules.encoder_names == ["text_encoder", "connectors"]
        assert modules.resident_names == ["vocoder"]
        assert len(modules.vaes) == 2

    def test_ltx23_pipeline_has_diffusion_pipeline_profiler_mixin(self):
        """LTX23Pipeline must support lightweight stage timing."""
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin

        assert issubclass(LTX23Pipeline, DiffusionPipelineProfilerMixin)


class TestLTX23DecodeConditioning:
    def test_decode_conditioning_expands_per_prompt_values_to_effective_batch(self):
        from vllm_omni.diffusion.models.ltx2_3.ltx2_3_latent_preparation import expand_per_prompt_decode_value

        assert expand_per_prompt_decode_value(
            [0.1, 0.2],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.1, 0.2, 0.2]
        assert expand_per_prompt_decode_value(
            [0.3],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.3, 0.3, 0.3, 0.3]
        assert expand_per_prompt_decode_value(
            [0.1, 0.2, 0.3, 0.4],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.2, 0.3, 0.4]

    def test_decode_conditioning_rejects_ambiguous_lengths(self):
        from vllm_omni.diffusion.models.ltx2_3.ltx2_3_latent_preparation import expand_per_prompt_decode_value

        with pytest.raises(ValueError, match="decode_timestep"):
            expand_per_prompt_decode_value(
                [0.1, 0.2, 0.3],
                prompt_batch_size=2,
                effective_batch_size=4,
                field_name="decode_timestep",
            )


class TestRegistryIntegration:
    """Verify all LTX-2.3 pipeline variants are registered."""

    def test_pipeline_models_registered(self):
        """LTX-2.3 pipeline variants must be in _DIFFUSION_MODELS."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_MODELS, f"{name} not found in _DIFFUSION_MODELS"

    def test_pipeline_module_paths(self):
        """Registry entries must point to the correct modules."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        # T2V -> pipeline_ltx2_3
        assert _DIFFUSION_MODELS["LTX23Pipeline"] == ("ltx2_3", "pipeline_ltx2_3", "LTX23Pipeline")

        # I2V -> pipeline_ltx2_3_image2video
        assert _DIFFUSION_MODELS["LTX23ImageToVideoPipeline"] == (
            "ltx2_3",
            "pipeline_ltx2_3_image2video",
            "LTX23ImageToVideoPipeline",
        )

    def test_post_process_funcs_registered(self):
        """Pipeline variants must map to get_ltx2_post_process_func."""
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_POST_PROCESS_FUNCS, f"{name} not in _DIFFUSION_POST_PROCESS_FUNCS"
            assert _DIFFUSION_POST_PROCESS_FUNCS[name] == "get_ltx2_post_process_func"

    def test_cache_dit_for_ltx2_does_not_have_custom_enablers_registered(self):
        """Pipeline variants are *not* registered in CUSTOM_DIT_ENABLERS."""
        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS

        # NOTE: We used to have custom enablers for this model, but refactored to handle
        # it more generically. Now we only need to ensure it has git cache adapter config.
        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name not in CUSTOM_DIT_ENABLERS, f"{name} not in CUSTOM_DIT_ENABLERS"


class TestVocoderSampleRateDetection:
    """Test _detect_vocoder_output_sample_rate logic."""

    def test_detects_48khz_from_config(self):
        """Should detect output_sampling_rate=48000 from vocoder/config.json."""
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000, "input_sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result == 48000

    def test_returns_none_for_no_output_sr(self):
        """Should return None if vocoder config has no output_sampling_rate."""
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result is None

    def test_returns_none_for_missing_directory(self):
        """Should return None if vocoder directory doesn't exist."""
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        result = _detect_vocoder_output_sample_rate("/nonexistent/path")
        assert result is None


class TestPostProcessFunction:
    """Test the post-process function factory."""

    def test_post_process_includes_audio_sample_rate(self):
        """Post-process func should include audio_sample_rate when detected."""
        import torch

        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import get_ltx2_post_process_func

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000}, f)

            # Create a minimal od_config mock
            class MockConfig:
                model = tmpdir

            func = get_ltx2_post_process_func(MockConfig())

            video = torch.zeros(1, 3, 4, 64, 64)
            audio = torch.zeros(1, 1, 48000)
            result = func((video, audio))

            assert isinstance(result, dict)
            assert "video" in result
            assert "audio" in result
            assert result["audio_sample_rate"] == 48000

    def test_post_process_without_vocoder_config(self):
        """Post-process func should work without vocoder config (no audio_sample_rate key)."""
        import torch

        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import get_ltx2_post_process_func

        class MockConfig:
            model = "/nonexistent/path"

        func = get_ltx2_post_process_func(MockConfig())

        video = torch.zeros(1, 3, 4, 64, 64)
        audio = torch.zeros(1, 1, 16000)
        result = func((video, audio))

        assert isinstance(result, dict)
        assert "video" in result
        assert "audio" in result
        assert "audio_sample_rate" not in result


class TestInitExports:
    """Test that __init__.py exports all LTX-2.3 classes."""

    def test_all_ltx23_classes_exported(self):
        """All LTX-2.3 pipeline classes must be in the ltx2_3 package __all__."""
        from vllm_omni.diffusion.models import ltx2_3

        expected_classes = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected_classes:
            assert hasattr(ltx2_3, name), f"{name} not exported from ltx2_3 package"
            assert name in ltx2_3.__all__, f"{name} not in ltx2_3.__all__"


class TestAudioLatentSPPadding:
    def test_sp_plan_shards_video_and_audio_timesteps(self):
        from vllm_omni.diffusion.models.ltx2_3.ltx2_3_transformer import LTX2VideoTransformer3DModel

        root_plan = LTX2VideoTransformer3DModel._build_sp_plan("split")[""]

        for name in ("timestep", "audio_timestep"):
            assert root_plan[name].split_dim == 1
            assert root_plan[name].expected_dims == 2
            assert not root_plan[name].split_output

    def test_prepare_video_latents_samples_packed_token_space(self):
        from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTX23Pipeline

        pipeline = _make_ltx23_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(0)

        latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=2,
            height=64,
            width=64,
            num_frames=9,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=generator,
        )

        direct_generator = torch.Generator(device="cpu").manual_seed(0)
        expected = torch.randn((1, 8, 2), generator=direct_generator, dtype=torch.float32)
        spatial_generator = torch.Generator(device="cpu").manual_seed(0)
        spatial_then_packed = LTX23Pipeline._pack_latents(
            torch.randn((1, 2, 2, 2, 2), generator=spatial_generator, dtype=torch.float32)
        )
        torch.testing.assert_close(latents, expected)
        assert not torch.equal(latents, spatial_then_packed)

    def test_prepare_audio_latents_samples_packed_token_space(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=2)
        generator = torch.Generator(device="cpu").manual_seed(0)

        latents, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=8,
            audio_latent_length=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=generator,
        )

        expected_generator = torch.Generator(device="cpu").manual_seed(0)
        expected = torch.randn((1, 4, 4), generator=expected_generator, dtype=torch.float32)
        assert original_num_frames == 3
        assert padded_num_frames == 4
        torch.testing.assert_close(latents, expected)

    def test_prepare_audio_latents_pads_generated_dummy_length_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=2)

        latents, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=8,
            num_mel_bins=64,
            audio_latent_length=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert original_num_frames == 1
        assert padded_num_frames == 2
        assert latents.shape == (1, 2, 128)

    def test_prepare_audio_latents_pads_provided_packed_sequence_dim_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(40, dtype=torch.float32).view(1, 10, 4)

        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=8,
            audio_latent_length=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 4)
        torch.testing.assert_close(padded[:, :10], latents)
        torch.testing.assert_close(padded[:, 10:], torch.zeros(1, 2, 4))

    def test_prepare_audio_latents_accepts_already_padded_4d_latents_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(96, dtype=torch.float32).view(1, 2, 12, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)
        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=16,
            audio_latent_length=audio_latent_length,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert audio_latent_length == 10
        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 8)
        torch.testing.assert_close(padded, pipeline._pack_audio_latents(latents))

    def test_audio_latent_patch_pack_roundtrips(self):
        pipeline = _make_ltx23_pipeline()
        latents = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)

        packed = pipeline._pack_audio_latents(latents, patch_size=2, patch_size_t=2)
        unpacked = pipeline._unpack_audio_latents(packed, latent_length=4, num_mel_bins=4, patch_size=2, patch_size_t=2)

        assert packed.shape == (1, 4, 8)
        torch.testing.assert_close(unpacked, latents)

    def test_resolve_audio_latent_length_preserves_legacy_4d_shape_inference(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 2, 13, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)

        assert audio_latent_length == 13

    def test_prepare_audio_latents_rejects_incompatible_provided_length(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 11, 4)

        with pytest.raises(ValueError, match="incompatible audio frame count"):
            pipeline.prepare_audio_latents(
                batch_size=1,
                num_channels_latents=2,
                num_mel_bins=8,
                audio_latent_length=10,
                dtype=torch.float32,
                device=torch.device("cpu"),
                latents=latents,
            )
