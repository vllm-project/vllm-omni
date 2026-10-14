# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AURA-compatible Qwen3-VL wrapper.

AURA ships Qwen3-VL weights with a repository-local ``configuration_qwen3_vl``
module in ``auto_map``. With ``trust_remote_code=True``, Transformers therefore
returns a remote ``Qwen3VLConfig`` class. The model structure is still Qwen3-VL,
but vLLM's stock multimodal processor checks for the upstream Transformers
``Qwen3VLConfig`` type exactly. This wrapper keeps the stock model and processor
logic, while accepting structurally-compatible remote Qwen3-VL configs.
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


class AuraQwen3VLProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Any:
        try:
            return self.ctx.get_hf_config(Qwen3VLConfig)
        except TypeError:
            config = self.ctx.get_hf_config()
            if (
                getattr(config, "model_type", None) == "qwen3_vl"
                and hasattr(config, "vision_config")
                and hasattr(config, "text_config")
            ):
                return config
            raise

    def get_hf_processor(self, **kwargs: object) -> Any:
        # AURA's config advertises `processing_qwen3_vl.py` in auto_map, but
        # the checkpoint snapshot does not include that file. Force the
        # upstream processor class while still allowing AURA's remote config.
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=AuraQwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AuraQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    pass
