# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recipe defaults for LTX-2.3 pipeline variants.

The current public pipelines implement the official one-stage recipe. Keep the
recipe data separate from the execution code so two-stage and HQ recipes can be
added without growing the base pipeline class further.
"""

from __future__ import annotations

from dataclasses import dataclass

LTX23_STAGE_2_DISTILLED_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)


@dataclass(frozen=True)
class LTX23GuidanceRecipe:
    cfg_scale: float
    stg_scale: float
    rescale_scale: float
    modality_scale: float
    skip_step: int
    stg_blocks: tuple[int, ...]


@dataclass(frozen=True)
class LTX23DenoiseStageRecipe:
    name: str
    width_scale: float
    height_scale: float
    num_inference_steps: int | None
    video_guidance: LTX23GuidanceRecipe | None
    audio_guidance: LTX23GuidanceRecipe | None
    sampler: str = "euler"
    sigma_values: tuple[float, ...] | None = None
    uses_distilled_lora: bool = False
    distilled_lora_strength: float | None = None
    uses_spatial_upsampler: bool = False


@dataclass(frozen=True)
class LTX23PipelineRecipe:
    name: str
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    video_guidance: LTX23GuidanceRecipe
    audio_guidance: LTX23GuidanceRecipe
    stages: tuple[LTX23DenoiseStageRecipe, ...]
    decode_timestep: float
    decode_noise_scale: float


LTX23_OFFICIAL_VIDEO_GUIDANCE = LTX23GuidanceRecipe(
    cfg_scale=3.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=(28,),
)

LTX23_OFFICIAL_AUDIO_GUIDANCE = LTX23GuidanceRecipe(
    cfg_scale=7.0,
    stg_scale=1.0,
    rescale_scale=0.7,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=(28,),
)

LTX23_HQ_VIDEO_GUIDANCE = LTX23GuidanceRecipe(
    cfg_scale=3.0,
    stg_scale=0.0,
    rescale_scale=0.45,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=(),
)

LTX23_HQ_AUDIO_GUIDANCE = LTX23GuidanceRecipe(
    cfg_scale=7.0,
    stg_scale=0.0,
    rescale_scale=1.0,
    modality_scale=3.0,
    skip_step=0,
    stg_blocks=(),
)


LTX23_ONE_STAGE_RECIPE = LTX23PipelineRecipe(
    name="one_stage_official",
    height=512,
    width=768,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=30,
    video_guidance=LTX23_OFFICIAL_VIDEO_GUIDANCE,
    audio_guidance=LTX23_OFFICIAL_AUDIO_GUIDANCE,
    stages=(
        LTX23DenoiseStageRecipe(
            name="stage_1",
            width_scale=1.0,
            height_scale=1.0,
            num_inference_steps=30,
            video_guidance=LTX23_OFFICIAL_VIDEO_GUIDANCE,
            audio_guidance=LTX23_OFFICIAL_AUDIO_GUIDANCE,
        ),
    ),
    decode_timestep=0.05,
    decode_noise_scale=0.025,
)

LTX23_TWO_STAGE_RECIPE = LTX23PipelineRecipe(
    name="two_stage_official",
    height=1024,
    width=1536,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=30,
    video_guidance=LTX23_OFFICIAL_VIDEO_GUIDANCE,
    audio_guidance=LTX23_OFFICIAL_AUDIO_GUIDANCE,
    stages=(
        LTX23DenoiseStageRecipe(
            name="stage_1",
            width_scale=0.5,
            height_scale=0.5,
            num_inference_steps=30,
            video_guidance=LTX23_OFFICIAL_VIDEO_GUIDANCE,
            audio_guidance=LTX23_OFFICIAL_AUDIO_GUIDANCE,
        ),
        LTX23DenoiseStageRecipe(
            name="stage_2_distilled_refine",
            width_scale=1.0,
            height_scale=1.0,
            num_inference_steps=None,
            video_guidance=None,
            audio_guidance=None,
            sigma_values=LTX23_STAGE_2_DISTILLED_SIGMAS,
            uses_distilled_lora=True,
            uses_spatial_upsampler=True,
        ),
    ),
    decode_timestep=0.05,
    decode_noise_scale=0.025,
)

LTX23_HQ_TWO_STAGE_RECIPE = LTX23PipelineRecipe(
    name="two_stage_hq_official",
    height=1088,
    width=1920,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=15,
    video_guidance=LTX23_HQ_VIDEO_GUIDANCE,
    audio_guidance=LTX23_HQ_AUDIO_GUIDANCE,
    stages=(
        LTX23DenoiseStageRecipe(
            name="stage_1_res2s",
            width_scale=0.5,
            height_scale=0.5,
            num_inference_steps=15,
            video_guidance=LTX23_HQ_VIDEO_GUIDANCE,
            audio_guidance=LTX23_HQ_AUDIO_GUIDANCE,
            sampler="res2s",
            uses_distilled_lora=True,
            distilled_lora_strength=0.25,
        ),
        LTX23DenoiseStageRecipe(
            name="stage_2_distilled_res2s_refine",
            width_scale=1.0,
            height_scale=1.0,
            num_inference_steps=None,
            video_guidance=None,
            audio_guidance=None,
            sampler="res2s",
            sigma_values=LTX23_STAGE_2_DISTILLED_SIGMAS,
            uses_distilled_lora=True,
            distilled_lora_strength=0.5,
            uses_spatial_upsampler=True,
        ),
    ),
    decode_timestep=0.05,
    decode_noise_scale=0.025,
)
