import os
from typing import Optional, Union
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .step_audio2_thinker import (
    StepAudio2MultiModalProcessor,
    StepAudio2ProcessingInfo,
    StepAudio2DummyInputsBuilder,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    StepAudio2MultiModalProcessor,
    info=StepAudio2ProcessingInfo,
    dummy_inputs=StepAudio2DummyInputsBuilder,
)
class StepAudio2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    Step-Audio2 Main Controller

    Manages two-stage inference pipeline:
    - Stage 1 (Thinker): Audio understanding and token generation
    - Stage 2 (Token2Wav): Audio token to waveform synthesis

    Usage:
        # Stage 1: Thinker
        model = StepAudio2ForConditionalGeneration(
            vllm_config=config,
            model_stage="thinker"
        )

        # Stage 2: Token2Wav
        model = StepAudio2ForConditionalGeneration(
            vllm_config=config,
            model_stage="token2wav"
        )
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.vllm_config = vllm_config

        # Determine which stage to load
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            # Initialize Thinker (LLM for audio understanding)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                architectures=["StepAudio2ThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.token2wav = None

            logger.info("Initialized Step-Audio2 Thinker (Stage 1)")

        elif self.model_stage == "token2wav":
            # Initialize Token2Wav (Audio synthesis)
            self.thinker = None
            self.token2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "token2wav"),
                hf_config=config,
                architectures=["StepAudio2Token2WavModel"],
            )
            self.model = self.token2wav

            logger.info("Initialized Step-Audio2 Token2Wav (Stage 2)")

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                f"Must be 'thinker' or 'token2wav'"
            )

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors
            if self.model_stage == "thinker"
            else lambda: None
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Get placeholder string for a modality"""
        if modality == "audio":
            return f"<|audio_{i}|>"
        return None

    def get_language_model(self) -> nn.Module:
        """Get the underlying language model"""
        if self.model_stage == "thinker":
            return self.thinker.get_language_model()
        else:
            return self.token2wav.get_language_model()

    def get_multimodal_embeddings(self, **kwargs):
        """Get multimodal embeddings - only used in Thinker stage"""
        if self.model_stage == "thinker":
            return self.thinker.get_multimodal_embeddings(**kwargs)
        return None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings"""
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through the model

        For Thinker:
            Returns hidden states/logits
        For Token2Wav:
            Returns waveform
        """
        return self.model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Compute logits from hidden states"""
        return self.model.compute_logits(hidden_states)

    def load_weights(self, weights):
        """Load weights"""
        return self.model.load_weights(weights)

    def move_submodules_to_devices(
        self,
        *,
        thinker_device: Optional[Union[str, torch.device]] = None,
        token2wav_device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Optionally move thinker/token2wav to different devices

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                token2wav_device='cuda:1',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(torch.device(thinker_device))
            logger.info(f"Moved Thinker to {thinker_device}")

        if token2wav_device is not None and self.token2wav is not None:
            self.token2wav.to(torch.device(token2wav_device))
            logger.info(f"Moved Token2Wav to {token2wav_device}")
