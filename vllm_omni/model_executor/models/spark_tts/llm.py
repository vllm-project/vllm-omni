# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen2 adapter for Omni's extra forward metadata.

All attention, logits and weight loading stay in vLLM's native Qwen2 model.
"""

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.model_executor.models.spark_tts.tokenizer import validate_tokenizer_vocab


class SparkQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if tokenizer is None:
            raise ValueError("Spark-TTS requires tokenizer initialization to validate its speech-code ids")
        validate_tokenizer_vocab(tokenizer.get_vocab())
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **omni_metadata: Any,
    ) -> torch.Tensor:
        del omni_metadata
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
