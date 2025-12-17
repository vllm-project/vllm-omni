"""
IndexTTS Stage 2: Vocoder - Mel-Spectrogram to Waveform
This stage takes mel-spectrograms as input and generates waveforms.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.model_executor.models.index_tts.index_tts_config import IndexTTSConfig
from vllm_omni.model_executor.models.index_tts.s2mel.modules.bigvgan import bigvgan

logger = init_logger(__name__)


class IndexTTSVocoderForConditionalGeneration(nn.Module, VllmModelForTextGeneration):
    """
    Stage 2: Vocoder model for generating waveforms from mel-spectrograms.
    Input: s2mel_spectrogram
    Output: waveforms
    """

    @classmethod
    def is_generative(cls) -> bool:
        return True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: IndexTTSConfig = vllm_config.model_config.hf_config
        self.prefix = prefix
        self.vocoder_name = self.config.vocoder.get("name")
        if not isinstance(self.vocoder_name, str) or not self.vocoder_name:
            raise RuntimeError("IndexTTS: vocoder.name must be provided for BigVGAN.from_pretrained")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        s2mel_spectrogram: torch.Tensor,  # [B, D, T_mel]
        **generation_kwargs,
    ):
        """
        Forward pass for vocoder-based waveform generation.
        """
        wav = self.bigvgan(s2mel_spectrogram.float()).squeeze().unsqueeze(0)
        wav = wav.squeeze(1)
        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
        return wav  # [B, T_audio]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for the vocoder model from the specified path.
        """
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "bigvgan.": "",
            }
        )
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(self.vocoder_name)
        self.bigvgan.remove_weight_norm()
        device = next(self.parameters()).device
        self.bigvgan.to(device)

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
