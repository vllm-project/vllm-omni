# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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


def test_ltx_versions_select_guidance_without_overriding_pipeline_control_flow():
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
