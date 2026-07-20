# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""User-visible one-stage defaults for the LTX model family."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LTXOneStageRecipe:
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    guidance_scale: float = 4.0


LTX2_ONE_STAGE_RECIPE = LTXOneStageRecipe()
LTX23_ONE_STAGE_RECIPE = LTXOneStageRecipe()
