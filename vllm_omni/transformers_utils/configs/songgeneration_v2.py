# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SongGeneration v2 config registration with transformers AutoConfig."""

from transformers import AutoConfig

from vllm_omni.model_executor.models.songgeneration_v2.configuration_songgeneration_v2 import (
    SongGenerationV2Config,
    SongGenerationV2Flow1dVAESeparateConfig,
    SongGenerationV2LeLMConfig,
)

AutoConfig.register("songgeneration_v2", SongGenerationV2Config)
AutoConfig.register("songgeneration_v2_lelm", SongGenerationV2LeLMConfig)
AutoConfig.register("songgeneration_v2_flow1dvae", SongGenerationV2Flow1dVAESeparateConfig)

__all__ = [
    "SongGenerationV2Config",
    "SongGenerationV2Flow1dVAESeparateConfig",
    "SongGenerationV2LeLMConfig",
]
