# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Distilled two-stage entry points for the LTX model family."""

from vllm_omni.diffusion.data import OmniDiffusionConfig

from .ltx2_components import LTX2_DISTILLED_COMPONENT_PROFILE
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_recipes import LTX2_DISTILLED_TWO_STAGE_RECIPE
from .pipeline_ltx2 import LTX2Pipeline


class LTX2DistilledPipeline(LTX2Pipeline):
    """Unified LTX-2 distilled two-stage T2V/I2V entry."""

    pipeline_kind = "distilled_two_stage"
    component_profile = LTX2_DISTILLED_COMPONENT_PROFILE
    pipeline_recipe = LTX2_DISTILLED_TWO_STAGE_RECIPE
    supports_request_batch = False
    support_image_input = True
    unified_text_image_entry = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
