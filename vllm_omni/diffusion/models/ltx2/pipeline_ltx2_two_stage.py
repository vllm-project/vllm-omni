# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-stage entry points for the LTX model family."""

from .ltx2_components import (
    LTX2_DISTILLED_COMPONENT_PROFILE,
    LTX2_TWO_STAGE_COMPONENT_PROFILE,
)
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_recipes import (
    LTX2_DISTILLED_TWO_STAGE_RECIPE,
    LTX2_TWO_STAGE_RECIPE,
)
from .pipeline_ltx2 import LTX2Pipeline


class LTX2TwoStagePipeline(LTX2Pipeline):
    """Regular checkpoint with low-resolution generation and LoRA refinement."""

    pipeline_kind = "two_stage"
    component_profile = LTX2_TWO_STAGE_COMPONENT_PROFILE
    pipeline_recipe = LTX2_TWO_STAGE_RECIPE
    supports_request_batch = False
    support_image_input = False


class LTX2DistilledPipeline(LTX2Pipeline):
    """Unified LTX-2 distilled two-stage T2V/I2V entry."""

    pipeline_kind = "distilled_two_stage"
    component_profile = LTX2_DISTILLED_COMPONENT_PROFILE
    pipeline_recipe = LTX2_DISTILLED_TWO_STAGE_RECIPE
    supports_request_batch = False
    support_image_input = True
    unified_text_image_entry = True
