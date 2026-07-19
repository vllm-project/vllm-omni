# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo config registration with transformers AutoConfig.

Registers ``Alpamayo15Config`` (model_type="alpamayo1_5") and
``AlpamayoR1Config`` (model_type="alpamayo_r1") so that
``AutoConfig.from_pretrained("/data/models/Alpamayo-1.5-10B")`` returns the
correct config class. Alpamayo-R1 checkpoints whose ``config.json`` already
declares ``model_type="qwen3_vl"`` continue to load as a plain Qwen3-VL config
and are routed by architecture name in the model registry.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.alpamayo.configuration_alpamayo import (
    Alpamayo15Config,
    AlpamayoR1Config,
)

AutoConfig.register("alpamayo1_5", Alpamayo15Config)
AutoConfig.register("alpamayo_r1", AlpamayoR1Config)

__all__ = ["Alpamayo15Config", "AlpamayoR1Config"]
