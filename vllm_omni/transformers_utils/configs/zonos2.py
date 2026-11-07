# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 config registration with transformers AutoConfig.

Registers Zonos2Config (model_type="zonos2") so that
``AutoConfig.from_pretrained(<converted checkpoint dir>)`` returns the correct
config class for the safetensors checkpoint produced by
``tools/convert_zonos2_to_safetensors.py``.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config

AutoConfig.register("zonos2", Zonos2Config)

__all__ = [
    "Zonos2Config",
]
