# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Confucius4-TTS config registration with Transformers AutoConfig."""

from transformers import AutoConfig

from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    Confucius4Config,
    Confucius4TTSConfig,
)

AutoConfig.register("confucius4", Confucius4Config)
AutoConfig.register("confucius4_tts", Confucius4TTSConfig)

__all__ = ["Confucius4Config", "Confucius4TTSConfig"]
