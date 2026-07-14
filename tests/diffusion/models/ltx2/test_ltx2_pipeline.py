# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the shared LTX pipeline runtime and public contracts."""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX2_COMPONENT_PROFILE,
    LTX23_COMPONENT_PROFILE,
)
from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import LTXI2VConditioningMixin
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXDenoiseExecutor, LTXPhaseResult
from vllm_omni.diffusion.models.ltx2.ltx2_guidance import (
    LTX_LEGACY_VELOCITY_GUIDANCE,
    LTX_OFFICIAL_X0_GUIDANCE,
)
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.ltx2_pipeline_base import LTXPipelineBase
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTX2_ONE_STAGE_RECIPE, LTX23_ONE_STAGE_RECIPE
from vllm_omni.diffusion.models.ltx2.ltx2_request import LTXRequestInputs
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import (
    LTX2ImageToVideoPipeline,
    LTX2Pipeline,
    LTX23ImageToVideoPipeline,
    LTX23Pipeline,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_two_stage import (
    LTX2ImageToVideoTwoStagesPipeline,
    LTX2TwoStagesPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ltx23_request_pipe(cls):
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 99
    return pipe


def _resolve_request_inputs_for_test(pipe, req):
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
        guidance_scale=4.0,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        max_sequence_length=None,
    )


def test_ltx_versions_share_runtime_without_cross_version_inheritance():
    assert issubclass(LTX2Pipeline, LTXPipelineBase)
    assert issubclass(LTX23Pipeline, LTXPipelineBase)
    assert not issubclass(LTX23Pipeline, LTX2Pipeline)
    assert LTX2Pipeline._pack_latents is LTX23Pipeline._pack_latents
    assert LTX2Pipeline.component_profile is LTX2_COMPONENT_PROFILE
    assert LTX23Pipeline.component_profile is LTX23_COMPONENT_PROFILE
    assert LTX2Pipeline.one_stage_recipe is LTX2_ONE_STAGE_RECIPE
    assert LTX23Pipeline.one_stage_recipe is LTX23_ONE_STAGE_RECIPE


def test_ltx_one_stage_variants_share_forward_template():
    assert LTX2Pipeline._forward_impl is LTXPipelineBase._forward_impl
    assert LTX2ImageToVideoPipeline._forward_impl is LTXPipelineBase._forward_impl
    assert LTX23Pipeline._forward_impl is LTXPipelineBase._forward_impl
    assert LTX23ImageToVideoPipeline._forward_impl is LTXPipelineBase._forward_impl
    assert LTX2Pipeline.forward is LTX23Pipeline.forward
    assert LTX2ImageToVideoPipeline.forward is LTX23ImageToVideoPipeline.forward


def test_ltx_versions_share_request_prompt_and_step_templates():
    shared_methods = (
        "_get_gemma_prompt_embeds",
        "encode_prompt",
        "check_inputs",
        "_resolve_request_inputs",
        "_prepare_prompt_context",
        "_denoise_step",
    )
    for method_name in shared_methods:
        base_method = getattr(LTXPipelineBase, method_name)
        assert getattr(LTX2Pipeline, method_name) is base_method
        assert getattr(LTX23Pipeline, method_name) is base_method


def test_ltx_versions_select_guidance_without_overriding_control_flow():
    assert LTX2Pipeline.guidance_strategy is LTX_LEGACY_VELOCITY_GUIDANCE
    assert LTX23Pipeline.guidance_strategy is LTX_OFFICIAL_X0_GUIDANCE
    assert LTX2Pipeline._predict_noise_for_step is LTXPipelineBase._predict_noise_for_step
    assert LTX23Pipeline._predict_noise_for_step is LTXPipelineBase._predict_noise_for_step
    assert LTX2Pipeline.combine_cfg_noise is LTXPipelineBase.combine_cfg_noise
    assert LTX23Pipeline.combine_cfg_noise is LTXPipelineBase.combine_cfg_noise


def test_ltx2_two_stage_variants_share_stage_orchestration():
    assert issubclass(LTX2ImageToVideoTwoStagesPipeline, LTX2TwoStagesPipeline)
    assert LTX2ImageToVideoTwoStagesPipeline.one_stage_pipeline_cls is LTX2ImageToVideoPipeline
    assert LTX2ImageToVideoTwoStagesPipeline._run_two_stage is LTX2TwoStagesPipeline._run_two_stage


def test_ltx_variants_share_denoise_loop_and_i2v_conditioning():
    assert "_denoise_loop" not in LTX2Pipeline.__dict__
    assert "_denoise_loop" not in LTX23Pipeline.__dict__
    assert issubclass(LTX2ImageToVideoPipeline, LTXI2VConditioningMixin)
    assert issubclass(LTX23ImageToVideoPipeline, LTXI2VConditioningMixin)
    assert LTX2ImageToVideoPipeline.prepare_latents is LTXI2VConditioningMixin.prepare_latents
    assert LTX23ImageToVideoPipeline.prepare_latents is LTXI2VConditioningMixin.prepare_latents
    assert LTX2Pipeline.prepare_latents is LTXPipelineBase.prepare_latents
    assert LTX23Pipeline.prepare_latents is LTXPipelineBase.prepare_latents
    assert LTX2Pipeline.prepare_audio_latents is LTXPipelineBase.prepare_audio_latents
    assert LTX23Pipeline.prepare_audio_latents is LTXPipelineBase.prepare_audio_latents
    assert LTX2Pipeline._decode_and_split is LTXPipelineBase._decode_and_split
    assert LTX23Pipeline._decode_and_split is LTXPipelineBase._decode_and_split


def test_denoise_executor_owns_progress_interrupt_and_current_timestep():
    updates = []

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self):
            updates.append(True)

    pipeline = SimpleNamespace(
        _current_timestep=None,
        interrupt=False,
        progress_bar=lambda total: Progress(),
    )
    seen: list[tuple[int, float, float]] = []
    timesteps = torch.tensor([3.0, 2.0, 1.0])
    initial_state = LTXAVState(video=torch.tensor(0.0), audio=torch.tensor(10.0))

    def step(index, timestep, state):
        seen.append((index, timestep.item(), pipeline._current_timestep.item()))
        pipeline.interrupt = True
        return LTXAVState(video=state.video + 1, audio=state.audio + 1)

    state = LTXDenoiseExecutor.run(pipeline, initial_state, timesteps, step)

    assert seen == [(0, 3.0, 3.0)]
    assert updates == [True]
    torch.testing.assert_close(state.video, torch.tensor(1.0))
    torch.testing.assert_close(state.audio, torch.tensor(11.0))


def test_ltx2_two_stage_reuses_prompt_context_between_phases():
    request_inputs = LTXRequestInputs(
        prompt="prompt",
        negative_prompt="negative",
        height=32,
        width=32,
        num_frames=1,
        frame_rate=24.0,
        num_inference_steps=4,
        guidance_scale=4.0,
        guidance_rescale=0.0,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        max_sequence_length=16,
    )
    prompt_context = object()
    phase_calls = []

    class FakePipe(torch.nn.Module):
        def _resolve_request_inputs(self, req, **kwargs):
            return request_inputs

        def _run_denoise_phase(self, req, inputs, *, prompt_context=None, **kwargs):
            phase_calls.append((inputs, prompt_context))
            if len(phase_calls) == 1:
                assert prompt_context is None
                context = prompt_context_sentinel
                video = torch.tensor([1.0])
                audio = torch.tensor([2.0])
            else:
                assert prompt_context is prompt_context_sentinel
                torch.testing.assert_close(inputs.latents, torch.tensor([11.0]))
                torch.testing.assert_close(inputs.audio_latents, torch.tensor([2.0]))
                assert inputs.guidance_scale == 1.0
                assert inputs.num_inference_steps == 3
                context = prompt_context
                video = torch.tensor([3.0])
                audio = torch.tensor([4.0])
            return LTXPhaseResult(
                forward_context=SimpleNamespace(prompt_context=context),
                video=video,
                audio=audio,
            )

        def _decode_and_split(self, forward_context, video, audio):
            return DiffusionOutput(output=(video, audio))

    class FakeUpsampler(torch.nn.Module):
        def forward(self, *, latents, output_type, return_dict):
            assert output_type == "latent"
            assert not return_dict
            return (latents + 10,)

    prompt_context_sentinel = prompt_context
    pipeline = object.__new__(LTX2TwoStagesPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.distilled = True
    pipeline.pipe = FakePipe()
    pipeline.upsample_pipe = FakeUpsampler()

    output = pipeline.forward(SimpleNamespace())

    assert len(phase_calls) == 2
    assert phase_calls[1][1] is prompt_context_sentinel
    torch.testing.assert_close(output.output[0], torch.tensor([3.0]))
    torch.testing.assert_close(output.output[1], torch.tensor([4.0]))


class TestLTX23RequestParsing:
    def test_t2v_and_i2v_share_request_input_resolution(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX23ImageToVideoPipeline, LTX23Pipeline
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
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX23Pipeline
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
    def test_encode_prompt_repeats_precomputed_embeds_for_num_outputs(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX23Pipeline

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
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX23Pipeline
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


class TestPipelineComponents:
    def test_ltx23_pipeline_declares_offload_components(self):
        """LTX23Pipeline must expose LTX-2.3-specific modules to offload discovery."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX23Pipeline
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


class TestLTX23DecodeConditioning:
    def test_decode_conditioning_expands_per_prompt_values_to_effective_batch(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import _expand_per_prompt_decode_value

        assert _expand_per_prompt_decode_value(
            [0.1, 0.2],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.1, 0.2, 0.2]
        assert _expand_per_prompt_decode_value(
            [0.3],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.3, 0.3, 0.3, 0.3]
        assert _expand_per_prompt_decode_value(
            [0.1, 0.2, 0.3, 0.4],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.2, 0.3, 0.4]

    def test_decode_conditioning_rejects_ambiguous_lengths(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import _expand_per_prompt_decode_value

        with pytest.raises(ValueError, match="decode_timestep"):
            _expand_per_prompt_decode_value(
                [0.1, 0.2, 0.3],
                prompt_batch_size=2,
                effective_batch_size=4,
                field_name="decode_timestep",
            )


class TestRegistryIntegration:
    """Verify all LTX-2.3 pipeline variants are registered."""

    def test_pipeline_module_paths(self):
        """Registry entries must point to the correct modules."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23Pipeline"] == ("ltx2", "pipeline_ltx2", "LTX23Pipeline")

        assert _DIFFUSION_MODELS["LTX23ImageToVideoPipeline"] == (
            "ltx2",
            "pipeline_ltx2",
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
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000, "input_sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result == 48000

    def test_returns_none_for_no_output_sr(self):
        """Should return None if vocoder config has no output_sampling_rate."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result is None

    def test_returns_none_for_missing_directory(self):
        """Should return None if vocoder directory doesn't exist."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import _detect_vocoder_output_sample_rate

        result = _detect_vocoder_output_sample_rate("/nonexistent/path")
        assert result is None


class TestPostProcessFunction:
    """Test the post-process function factory."""

    def test_post_process_includes_audio_sample_rate(self):
        """Post-process func should include audio_sample_rate when detected."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import get_ltx2_post_process_func

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

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import get_ltx2_post_process_func

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
