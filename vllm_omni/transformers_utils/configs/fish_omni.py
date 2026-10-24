# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fish Audio Drama 3 config and quantization registration.

Registers FishOmniConfig (model_type="fish_omni") with transformers AutoConfig
and the checkpoint's ``bsgemm-moe`` quantization method with vLLM.
"""

from transformers import AutoConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForCausalLMConfig

from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniConfig
from vllm_omni.model_executor.models.fish_omni.quantization import ExpertFp8Config  # noqa: F401

AutoConfig.register("fish_omni", FishOmniConfig)
# The Qwen3.5 backbone keeps its gated-delta-net state in the checkpoint's
# mamba_ssm_dtype (float32); vLLM applies that per architecture name.
MODELS_CONFIG_MAP.setdefault("FishOmniForConditionalGeneration", Qwen3_5ForCausalLMConfig)

__all__ = ["FishOmniConfig"]
