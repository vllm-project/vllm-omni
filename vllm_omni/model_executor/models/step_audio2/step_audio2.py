import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .step_audio2_thinker import (
    StepAudio2DummyInputsBuilder,
    StepAudio2MultiModalProcessor,
    StepAudio2ProcessingInfo,
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
    - Stage 0 (Thinker): Audio understanding and token generation
    - Stage 1 (Token2Wav): Audio token to waveform synthesis

    Usage:
        # Stage 0: Thinker
        model = StepAudio2ForConditionalGeneration(
            vllm_config=config,
            model_stage="thinker"
        )

        # Stage 1: Token2Wav
        model = StepAudio2ForConditionalGeneration(
            vllm_config=config,
            model_stage="token2wav"
        )
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.vllm_config = vllm_config

        raw_model_stage = vllm_config.model_config.model_stage
        # Normalise: "step_audio2_thinker" → "thinker"
        self.model_stage = "thinker" if raw_model_stage in ("thinker", "step_audio2_thinker") else raw_model_stage

        if self.model_stage == "thinker":
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                architectures=["StepAudio2ThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.token2wav = None

            logger.info("Initialized Step-Audio2 Thinker (Stage 0)")

        elif self.model_stage == "token2wav":
            self.thinker = None
            self.token2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "token2wav"),
                hf_config=config,
                architectures=["StepAudio2Token2WavForConditionalGeneration"],
            )
            self.model = self.token2wav

            logger.info("Initialized Step-Audio2 Token2Wav (Stage 1)")

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. Must be 'thinker'/'step_audio2_thinker' or 'token2wav'"
            )

        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.model_stage == "thinker" else lambda: None
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Get placeholder string for a modality

        Returns:
            For audio: "<audio_patch>" (matches processor's audio_token)
        """
        if modality == "audio":
            return "<audio_patch>"
        return None

    def get_language_model(self) -> nn.Module:
        """Get the underlying language model."""
        if self.model_stage == "thinker":
            return self.thinker.get_language_model()
        # Token2Wav is not a real language model. Return the wrapper itself so
        # vLLM interface helpers resolve embed_input_ids() on the adapter layer.
        return self.token2wav

    def get_multimodal_embeddings(self, **kwargs):
        """Get multimodal embeddings - only used in Thinker stage."""
        if self.model_stage == "thinker":
            return self.thinker.get_multimodal_embeddings(**kwargs)
        return None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compatibility helper used by older call sites."""
        if self.model_stage == "thinker":
            return self.model.get_input_embeddings(input_ids)
        return self.model.embed_input_ids(input_ids)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        """Explicit vLLM embedding hook for both stages."""
        if self.model_stage == "token2wav":
            return self.model.embed_input_ids(input_ids)
        return self.model.get_input_embeddings(input_ids, multimodal_embeddings)

    def embed_multimodal(self, **kwargs):
        """Delegate multimodal embedding to thinker stage only."""
        if self.model_stage == "thinker":
            return self.model.embed_multimodal(**kwargs)
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
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
            **kwargs,
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
        thinker_device: str | torch.device | None = None,
        token2wav_device: str | torch.device | None = None,
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
