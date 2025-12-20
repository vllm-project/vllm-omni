from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class CosyVoiceModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_stage = vllm_config.model_config.model_stage
        self.model = None
        if self.model_stage == "text_speech_lm":
            # Initialize text to speech LM stage
            self.text_speech_lm_model = None  # Replace with actual model initialization (e.g.,
            self.model = self.text_speech_lm_model

        elif self.model_stage == "chunk_aware_flow_matching":
            # Initialize chunk aware flow matching stage
            self.chunk_aware_flow_matching_model = None  # Replace with actual model initialization
            self.model = self.chunk_aware_flow_matching_model
            pass
        elif self.model_stage == "acoustic_features_to_waveform":
            # Initialize acoustic features to waveform stage
            self.acoustic_features_to_waveform_model = None  # Replace with actual model initialization
            self.model = self.acoustic_features_to_waveform_model
            pass
        else:
            raise ValueError(f"Unknown model stage: {self.model_stage}")

    def forward(self):
        pass

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "text_speech_lm":
            # Load weights for text to speech LM stage
            pass
        elif self.model_stage == "chunk_aware_flow_matching":
            # Load weights for chunk aware flow matching stage
            pass
        elif self.model_stage == "acoustic_features_to_waveform":
            # Load weights for acoustic features to waveform stage
            pass
        else:
            raise ValueError(f"Unknown model stage: {self.model_stage}")
