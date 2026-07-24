# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Distilled two-stage entry points for the LTX model family."""

import os

from vllm_omni.diffusion.data import OmniDiffusionConfig

from .ltx2_components import LTX2_COMPONENT_PROFILE
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_recipes import LTX2_DISTILLED_TWO_STAGE_RECIPE
from .pipeline_ltx2 import LTX2Pipeline
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline


class LTX2DistilledPipeline(LTX2Pipeline):
    """Unified LTX-2 distilled two-stage T2V/I2V entry."""

    pipeline_kind = "distilled_two_stage"
    component_profile = LTX2_COMPONENT_PROFILE
    pipeline_recipe = LTX2_DISTILLED_TWO_STAGE_RECIPE
    supports_request_batch = False
    support_image_input = True
    unified_text_image_entry = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        model_path = od_config.model
        self.distilled = "distilled" in os.path.basename(os.path.normpath(model_path))
        if not self.distilled:
            raise NotImplementedError(f"{model_path} is not supported for {self.__class__.__name__}.")

        super().__init__(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(vae=self.vae, od_config=od_config)
