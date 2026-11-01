# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM-Omni wrapper for the native FunAudioChat multimodal model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .common import register_funaudiochat_processor

try:
    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatForConditionalGeneration as VllmFunAudioChatForConditionalGeneration,
    )
except ImportError:
    VllmFunAudioChatForConditionalGeneration = None


@register_funaudiochat_processor
class FunAudioChatForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """FunAudioChat model wrapper backed by vLLM's native audio/Qwen3 model."""

    supports_multimodal = True
    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = False
    input_modalities = "audio"
    have_multimodal_outputs = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if VllmFunAudioChatForConditionalGeneration is None:
            raise ImportError("FunAudioChat requires a vLLM version with vllm.model_executor.models.funaudiochat.")

        self.model = VllmFunAudioChatForConditionalGeneration(
            vllm_config=vllm_config,
            prefix=prefix,
        )
        self.config = self.model.config
        self.multimodal_config = self.model.multimodal_config
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if VllmFunAudioChatForConditionalGeneration is None:
            raise ImportError(
                "FunAudioChat requires a vLLM version with "
                "vllm.model_executor.models.funaudiochat."
            )
        return VllmFunAudioChatForConditionalGeneration.get_placeholder_str(modality, i)

    def embed_multimodal(self, **kwargs: object):
        return self.model.embed_multimodal(**kwargs)

    def get_multimodal_embeddings(self, **kwargs: object):
        return self.embed_multimodal(**kwargs)

    def get_language_model(self) -> nn.Module:
        return self.model.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs={})

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)
