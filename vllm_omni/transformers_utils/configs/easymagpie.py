# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from transformers import AutoConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

from vllm_omni.model_executor.models.easymagpie.codec.config import EasyMagpieCodecConfig

AutoConfig.register(EasyMagpieCodecConfig.model_type, EasyMagpieCodecConfig)
MODELS_CONFIG_MAP.setdefault("EasyMagpieCodecForConditionalGeneration", MambaModelConfig)
