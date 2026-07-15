# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""User-visible one-stage defaults for the LTX model family."""

from dataclasses import dataclass

from .ltx2_guidance import LTXGuidanceSpec, LTXModalityGuidance


@dataclass(frozen=True)
class LTXOneStageRecipe:
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    guidance: LTXGuidanceSpec = LTXGuidanceSpec()
    use_official_sigma_schedule: bool = True


def _official_guidance(stg_block: int) -> LTXGuidanceSpec:
    return LTXGuidanceSpec(
        video=LTXModalityGuidance(
            cfg_scale=3.0,
            stg_scale=1.0,
            modality_scale=3.0,
            rescale_scale=0.7,
            stg_blocks=(stg_block,),
        ),
        audio=LTXModalityGuidance(
            cfg_scale=7.0,
            stg_scale=1.0,
            modality_scale=3.0,
            rescale_scale=0.7,
            stg_blocks=(stg_block,),
        ),
    )


LTX2_ONE_STAGE_RECIPE = LTXOneStageRecipe(guidance=_official_guidance(29))
LTX23_ONE_STAGE_RECIPE = LTXOneStageRecipe(
    num_inference_steps=30,
    guidance=_official_guidance(28),
)
LTX_POSITIVE_ONLY_RECIPE = LTXOneStageRecipe(
    guidance=LTXGuidanceSpec.positive_only(),
    use_official_sigma_schedule=False,
)
