# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AURA-compatible Qwen3.5-VL wrapper (AURA_v2 / qwen3_5 checkpoints).

Mirrors ``aura_omni/qwen3_vl.py``: keep stock vLLM Qwen3.5 multimodal logic while
accepting structurally compatible remote configs. Stage1 → TTS only needs finished
text / token ids, so full-prefix hidden D2H is left off.
"""

from __future__ import annotations

from typing import Any

from transformers import Qwen3_5Config
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


class AuraQwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_config(self) -> Any:
        try:
            return self.ctx.get_hf_config(Qwen3_5Config)
        except TypeError:
            config = self.ctx.get_hf_config()
            if (
                getattr(config, "model_type", None) == "qwen3_5"
                and hasattr(config, "vision_config")
                and hasattr(config, "text_config")
            ):
                return config
            raise

    def get_hf_processor(self, **kwargs: object) -> Any:
        # Qwen3.5 multimodal still uses the Qwen3-VL processor class in vLLM.
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=AuraQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AuraQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # Stage1 AURA → TTS only needs finished text / token ids.
    # Hybrid GDN + attention has more than one KV group, so the omni
    # single-group output cache does not apply. Hits reuse vLLM's own KV
    # and, with mamba_cache_mode=align, the recurrent state.
    requires_full_prefix_cached_hidden_states = False
    uses_native_kv_cache_only = True
