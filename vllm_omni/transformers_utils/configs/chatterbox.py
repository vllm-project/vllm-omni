# SPDX-License-Identifier: Apache-2.0
"""Register Chatterbox TTS configs with HuggingFace AutoConfig.

Chatterbox does not ship a standard config.json, so we register our custom
config classes here.  The side-effect import in ``__init__.py`` ensures this
runs at package load time.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.chatterbox.configuration_chatterbox import (
    ChatterboxConfig,
    ChatterboxTurboConfig,
)

AutoConfig.register("chatterbox_turbo", ChatterboxTurboConfig)
AutoConfig.register("chatterbox", ChatterboxConfig)
