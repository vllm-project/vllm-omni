# Copyright 2025 vLLM-Omni Team
"""Top-level dispatcher for Kimi Audio model."""

import os
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.kimi_audio import (
    KimiAudioDummyInputsBuilder,
    KimiAudioProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.kimi_audio.custom_processor import (
    CustomKimiAudioMultiModalProcessor,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


@MULTIMODAL_REGISTRY.register_processor(
    CustomKimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class KimiAudioForConditionalGeneration(nn.Module):
    """Top-level model that dispatches to stage 0 (LLM) or stage 1 (detokenizer).

    The stage is determined by the MODEL_STAGE environment variable:
    - "fused_llm" (default): Stage 0 - LLM with dual output heads
    - "audio_detokenizer": Stage 1 - Flow-matching detokenizer + vocoder
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix

        # Allow the runner to copy per-request additional_information into the
        # model_intermediate_buffer so make_omni_output/sample can see task_type.
        self.enable_update_additional_information = True

        # Check model_stage env var (set by stage runner)
        model_stage = os.environ.get("MODEL_STAGE", "fused_llm")

        if model_stage == "fused_llm":
            # Stage 0: LLM with dual output heads
            from .kimi_audio_llm import KimiAudioLLMForConditionalGeneration

            self.model = KimiAudioLLMForConditionalGeneration(vllm_config=vllm_config, prefix=prefix)
            # The Kimi Audio checkpoint stores the Whisper encoder weights in the
            # whisper-large-v3 subfolder rather than the main safetensors.
            self.secondary_weights = [
                DefaultModelLoader.Source(
                    model_or_path=vllm_config.model_config.model,
                    subfolder="whisper-large-v3",
                    revision=vllm_config.model_config.revision,
                )
            ]
        elif model_stage == "audio_detokenizer":
            # Stage 1: Flow-matching detokenizer + vocoder
            from .kimi_audio_detokenizer import KimiAudioDetokenizerForConditionalGeneration

            self.model = KimiAudioDetokenizerForConditionalGeneration(vllm_config=vllm_config, prefix=prefix)
        else:
            raise ValueError(f"Unknown MODEL_STAGE: {model_stage}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        multimodal_embeddings: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Forward pass - delegates to the appropriate stage model."""
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            multimodal_embeddings=multimodal_embeddings,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: any,
    ) -> torch.Tensor | None:
        """Compute logits - delegates to the appropriate stage model."""
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return None

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor] | None:
        """Process multimodal inputs - delegates to the appropriate stage model."""
        if hasattr(self.model, "embed_multimodal"):
            return self.model.embed_multimodal(**kwargs)
        return None

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load weights - delegates to the appropriate stage model."""
        self.model.load_weights(weights)
