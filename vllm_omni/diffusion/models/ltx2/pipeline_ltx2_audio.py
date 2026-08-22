# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unified full-model text-to-audio entry point for the LTX family."""

from typing import ClassVar

from .ltx2_audio_runtime import LTXAudioRuntime
from .ltx2_components import LTX2_T2A_COMPONENT_PROFILE
from .ltx2_components import (
    get_ltx2_audio_post_process_func as get_ltx2_audio_post_process_func,  # noqa: F401
)
from .ltx2_recipes import LTX2_T2A_RECIPE


class LTX2TextToAudioPipeline(LTXAudioRuntime):
    """Generate audio without constructing or denoising a video branch."""

    pipeline_kind = "text_to_audio"
    component_profile = LTX2_T2A_COMPONENT_PROFILE
    pipeline_recipe = LTX2_T2A_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)
