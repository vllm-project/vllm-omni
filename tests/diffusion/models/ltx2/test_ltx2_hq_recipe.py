# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Declarative contract tests for the LTX-2.3/2.5 two-stage HQ entry."""

import pytest

from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX23_TWO_STAGE_COMPONENT_PROFILE,
    LTX25_TWO_STAGE_COMPONENT_PROFILE,
    resolve_ltx_checkpoint_kind,
    resolve_ltx_component_profile,
)
from vllm_omni.diffusion.models.ltx2.ltx2_guidance import LTXGuidancePlan, LTXGuidanceSpec
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX23_TWO_STAGE_HQ_RECIPE,
    LTX23_TWO_STAGE_RECIPE,
    LTX25_DEFAULT_NEGATIVE_PROMPT,
    LTX25_TWO_STAGE_HQ_RECIPE,
    LTX_STAGE_2_DISTILLED_SIGMAS,
    LTXPhaseRecipe,
    resolve_ltx_pipeline_recipe,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_two_stage import LTX2TwoStageHQPipeline
from vllm_omni.model_extras import get_extra_body_params, get_extra_output_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("recipe", [LTX23_TWO_STAGE_HQ_RECIPE, LTX25_TWO_STAGE_HQ_RECIPE])
def test_ltx_hq_recipe_matches_official_phase_contract(recipe):
    stage1, stage2 = recipe.phases

    assert (recipe.width, recipe.height) == (1920, 1088)
    assert recipe.num_frames == 121
    assert recipe.frame_rate == 24.0
    assert recipe.num_inference_steps == 15
    assert (recipe.video_output_phase, recipe.audio_output_phase) == (1, 0)
    assert not recipe.allow_request_sigmas
    assert recipe.allow_request_phase_sigmas
    assert not recipe.allow_request_latents

    assert stage1.name == "generate_lowres"
    assert stage1.spatial_downscale == 2
    assert stage1.noise_scale == 1.0
    assert stage1.sampler == "res2s"
    assert stage1.sigmas is None
    assert not stage1.use_official_sigma_schedule
    assert stage1.use_latent_dependent_sigma_schedule
    assert (stage1.adapter_slot, stage1.adapter_strength) == ("ltx_distilled", 0.25)
    assert LTXGuidancePlan.build(stage1.guidance).names == ("cond", "uncond", "mod")
    assert (
        stage1.guidance.video.cfg_scale,
        stage1.guidance.video.stg_scale,
        stage1.guidance.video.modality_scale,
        stage1.guidance.video.rescale_scale,
    ) == (3.0, 0.0, 3.0, 0.45)
    assert (
        stage1.guidance.audio.cfg_scale,
        stage1.guidance.audio.stg_scale,
        stage1.guidance.audio.modality_scale,
        stage1.guidance.audio.rescale_scale,
    ) == (7.0, 0.0, 3.0, 1.0)

    assert stage2.name == "refine"
    assert stage2.input_transform == "spatial_upsample"
    assert stage2.sampler == "res2s"
    assert stage2.guidance == LTXGuidanceSpec.positive_only()
    assert not stage2.allow_guidance_override
    assert stage2.sigmas == LTX_STAGE_2_DISTILLED_SIGMAS
    assert stage2.num_inference_steps == 3
    assert (stage2.adapter_slot, stage2.adapter_strength) == ("ltx_distilled", 0.5)


def test_ltx25_hq_keeps_generation_specific_negative_prompt():
    assert LTX25_TWO_STAGE_HQ_RECIPE.negative_prompt == LTX25_DEFAULT_NEGATIVE_PROMPT
    assert LTX23_TWO_STAGE_HQ_RECIPE.negative_prompt != LTX25_DEFAULT_NEGATIVE_PROMPT


def test_ltx_hq_entry_supports_ltx23_and_ltx25():
    assert resolve_ltx_checkpoint_kind("two_stage_hq") == "regular"
    assert resolve_ltx_component_profile("two_stage_hq", "2.3") is LTX23_TWO_STAGE_COMPONENT_PROFILE
    assert resolve_ltx_component_profile("two_stage_hq", "2.5") is LTX25_TWO_STAGE_COMPONENT_PROFILE
    assert resolve_ltx_pipeline_recipe("two_stage_hq", "2.3") is LTX23_TWO_STAGE_HQ_RECIPE
    assert resolve_ltx_pipeline_recipe("two_stage_hq", "2.5") is LTX25_TWO_STAGE_HQ_RECIPE
    with pytest.raises(ValueError, match="Unsupported LTX component kind/version"):
        resolve_ltx_component_profile("two_stage_hq", "2")
    with pytest.raises(ValueError, match="Unsupported LTX pipeline kind/version"):
        resolve_ltx_pipeline_recipe("two_stage_hq", "2")

    assert LTX2TwoStageHQPipeline.pipeline_kind == "two_stage_hq"
    assert LTX2TwoStageHQPipeline.support_image_input
    assert LTX2TwoStageHQPipeline.unified_text_image_entry
    assert not LTX2TwoStageHQPipeline.supports_request_batch


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"adapter_slot": "ltx_distilled"}, "slot and strength"),
        ({"adapter_strength": 0.25}, "slot and strength"),
        ({"adapter_slot": "ltx_distilled", "adapter_strength": float("nan")}, "finite"),
        ({"adapter_slot": "ltx_distilled", "adapter_strength": float("inf")}, "finite"),
        ({"sampler": "unknown"}, "sampler"),
        (
            {"use_official_sigma_schedule": True, "use_latent_dependent_sigma_schedule": True},
            "both official and latent-dependent",
        ),
        (
            {
                "sigmas": (1.0, 0.0),
                "use_official_sigma_schedule": False,
                "use_latent_dependent_sigma_schedule": True,
            },
            "explicit sigmas",
        ),
    ],
)
def test_ltx_phase_recipe_validates_sampler_schedule_and_adapter(kwargs, error):
    with pytest.raises(ValueError, match=error):
        LTXPhaseRecipe(name="invalid", guidance=LTXGuidanceSpec.positive_only(), **kwargs)


def test_existing_two_stage_recipe_keeps_euler_and_unit_strength():
    stage1, stage2 = LTX23_TWO_STAGE_RECIPE.phases

    assert stage1.sampler == stage2.sampler == "euler"
    assert not stage1.use_latent_dependent_sigma_schedule
    assert not stage2.use_latent_dependent_sigma_schedule
    assert (stage1.adapter_slot, stage1.adapter_strength) == (None, None)
    assert (stage2.adapter_slot, stage2.adapter_strength) == ("ltx_distilled", 1.0)


def test_ltx_hq_registry_and_request_extras():
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _DIFFUSION_POST_PROCESS_FUNCS

    assert _DIFFUSION_MODELS["LTX2TwoStageHQPipeline"] == (
        "ltx2",
        "pipeline_ltx2_two_stage",
        "LTX2TwoStageHQPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["LTX2TwoStageHQPipeline"] == "get_ltx2_post_process_func"
    assert get_extra_body_params("LTX2TwoStageHQPipeline") == get_extra_body_params("LTX2TwoStagePipeline")
    assert get_extra_output_params("LTX2TwoStageHQPipeline") == frozenset()
