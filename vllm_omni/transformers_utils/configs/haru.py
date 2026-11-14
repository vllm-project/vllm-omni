# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Haru-4B config registration.

Registers HaruConfig (model_type="haru") with transformers AutoConfig.
"""

from transformers import AutoConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForCausalLMConfig

from vllm_omni.model_executor.models.haru.configuration_haru import HaruConfig

AutoConfig.register("haru", HaruConfig)
# The backbone is a text-only Qwen3.5: same hybrid-cache and position setup, and
# vLLM keeps the gated-delta-net state in the checkpoint's mamba_ssm_dtype.
MODELS_CONFIG_MAP.setdefault("HaruForConditionalGeneration", Qwen3_5ForCausalLMConfig)

__all__ = ["HaruConfig"]
