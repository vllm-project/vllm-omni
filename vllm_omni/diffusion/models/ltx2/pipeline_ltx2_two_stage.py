# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-stage entry points for the LTX model family."""

from typing import ClassVar

from .ltx2_components import (
    LTX2_DISTILLED_COMPONENT_PROFILE,
    LTX2_TWO_STAGE_COMPONENT_PROFILE,
)
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_recipes import (
    LTX2_DISTILLED_TWO_STAGE_RECIPE,
    LTX2_TWO_STAGE_RECIPE,
)
from .ltx2_runtime import LTXRuntime


class _LTX2TwoStageBase(LTXI2VConditioningMixin, LTXRuntime):
    """Static discovery contract shared by the two-stage public entries."""

    _dit_modules: ClassVar[list[str]] = list(LTX2_TWO_STAGE_COMPONENT_PROFILE.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(LTX2_TWO_STAGE_COMPONENT_PROFILE.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(LTX2_TWO_STAGE_COMPONENT_PROFILE.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(LTX2_TWO_STAGE_COMPONENT_PROFILE.resident_modules)
    supports_request_batch = False
    support_image_input = True
    unified_text_image_entry = True


class LTX2TwoStagePipeline(_LTX2TwoStageBase):
    """Regular checkpoint with low-resolution generation and LoRA refinement."""

    pipeline_kind = "two_stage"
    component_profile = LTX2_TWO_STAGE_COMPONENT_PROFILE
    pipeline_recipe = LTX2_TWO_STAGE_RECIPE


class LTX2DistilledTwoStagePipeline(_LTX2TwoStageBase):
    """Merged-distilled checkpoint with latent upsampling and refinement."""

    pipeline_kind = "distilled_two_stage"
    component_profile = LTX2_DISTILLED_COMPONENT_PROFILE
    pipeline_recipe = LTX2_DISTILLED_TWO_STAGE_RECIPE


# Backwards-compatible name retained for existing LTX-2/LTX-2.3 recipes.
LTX2DistilledPipeline = LTX2DistilledTwoStagePipeline
