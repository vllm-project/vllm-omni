# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recipe defaults for LTX-2.3 pipeline variants.

The current public pipelines implement the official one-stage recipe. Keep the
recipe data separate from the execution code so two-stage and HQ recipes can be
added without growing the base pipeline class further.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    stages: tuple[LTX23DenoiseStageRecipe, ...]
    decode_timestep: float
    decode_noise_scale: float

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError(f"LTX-2.3 recipe {self.name!r} must define an initial denoise stage.")
        initial_stage = self.initial_denoise_stage
        if initial_stage.num_inference_steps is None:
            raise ValueError(f"LTX-2.3 recipe {self.name!r} initial stage must define num_inference_steps.")
        if initial_stage.video_guidance is None or initial_stage.audio_guidance is None:
            raise ValueError(f"LTX-2.3 recipe {self.name!r} initial stage must define video and audio guidance.")

    @property
    def initial_denoise_stage(self) -> LTX23DenoiseStageRecipe:
        """The stage that supplies request defaults and begins generation."""
        return self.stages[0]


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

LTX23_ONE_STAGE_RECIPE = LTX23PipelineRecipe(
    name="one_stage_official",
    height=512,
    width=768,
    num_frames=121,
    frame_rate=24.0,
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
