# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native vLLM registration for Microsoft's Mage-VL architecture.

Mage-VL uses the Qwen3 language backbone and a Qwen2-VL-compatible multimodal
input contract.  The executor deliberately reuses vLLM's optimized Qwen3-VL
implementation; the compatibility processing info accepts Mage's remote
``MageVLConfig`` while preserving image/video placeholder semantics.
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


class MageVLProcessingInfo(Qwen3VLProcessingInfo):
    """Accept Mage's structurally compatible remote configuration."""

    def get_hf_config(self) -> Any:
        try:
            return self.ctx.get_hf_config(Qwen3VLConfig)
        except TypeError:
            config = self.ctx.get_hf_config()
            if (
                getattr(config, "model_type", None) == "mage_vl"
                and hasattr(config, "vision_config")
                and hasattr(config, "text_config")
            ):
                return config
            raise

    def get_hf_processor(self, **kwargs: object) -> Any:
        # Mage's processor config points at remote ``MageVLVideoProcessor``
        # code that is not an AutoProcessor class.  Build the maintained
        # Qwen3-VL processor from its concrete local components instead.
        from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLVideoProcessor

        model_path = self.ctx.model_config.model
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
        video_processor = Qwen2VLVideoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        return Qwen3VLProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=MageVLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class MageVLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """vLLM executor registered for ``MageVLForConditionalGeneration``."""

    pass
