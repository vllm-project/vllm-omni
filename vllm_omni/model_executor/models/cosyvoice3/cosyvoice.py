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

        if self.model_stage == "text_speech_lm":
            # Initialize text to speech LM stage
            pass

        elif self.model_stage == "chunk_aware_flow_matching":
            # Initialize chunk aware flow matching stage
            pass
        elif self.model_stage == "acoustic_features_to_waveform":
            # Initialize acoustic features to waveform stage
            pass
        else:
            raise ValueError(f"Unknown model stage: {self.model_stage}")

    def forward(self):
        pass

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        pass
