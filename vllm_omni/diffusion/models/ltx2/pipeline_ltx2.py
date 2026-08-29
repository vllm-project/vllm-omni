# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One-stage entry points for the LTX model family."""

from __future__ import annotations

from typing import ClassVar

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin

from .ltx2_components import LTX2_COMPONENT_PROFILE, LTX2_DISTILLED_ONE_STAGE_COMPONENT_PROFILE
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_recipes import LTX2_DISTILLED_ONE_STAGE_RECIPE, LTX2_ONE_STAGE_RECIPE, LTX_POSITIVE_ONLY_RECIPE
from .ltx2_runtime import LTXRuntime


class LTX2Pipeline(LTXI2VConditioningMixin, LTXRuntime):
    """LTX-2 family one-stage entry, configured from checkpoint metadata."""

    pipeline_kind = "one_stage"
    unified_text_image_entry = True
    component_profile = LTX2_COMPONENT_PROFILE
    pipeline_recipe = LTX2_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)
    supports_request_batch = True


class LTX2DistilledOneStagePipeline(LTX2Pipeline):
    """Merged-distilled checkpoint at its native one-stage resolution."""

    pipeline_kind = "distilled_one_stage"
    component_profile = LTX2_DISTILLED_ONE_STAGE_COMPONENT_PROFILE
    pipeline_recipe = LTX2_DISTILLED_ONE_STAGE_RECIPE


class LTX2T2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX2 T2V entry for FastGen DMD2-distilled models."""

    pipeline_kind = "dmd2"
    support_image_input = False
    pipeline_recipe = LTX_POSITIVE_ONLY_RECIPE

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()


class LTX2I2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX2 I2V entry for FastGen DMD2-distilled models."""

    pipeline_kind = "dmd2"
    unified_text_image_entry = False
    pipeline_recipe = LTX_POSITIVE_ONLY_RECIPE

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
