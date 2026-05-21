# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# MiniMind-O Code2Wav stage - Mimi codec decoder.

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.minimind_o.config import MiniMindOCode2WavConfig

logger = init_logger(__name__)


class MiniMindOCode2Wav(nn.Module):
    """
    Mimi codec decoder - converts 8-layer Mimi codes to 24kHz waveform.

    Specs:
    - 8-layer codebook
    - 12.5 Hz frame rate
    - 24 kHz output sample rate
    - Upsampling factor: ~1920x (24000 / 12.5)

    Note: This is a placeholder implementation. The actual Mimi decoder
    architecture needs to be obtained from the MiniMind-O repository.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "code2wav.": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MiniMindOCode2WavConfig = vllm_config.model_config.hf_config
        self.config = config
        self.prefix = prefix

        # Code embedding for 8-layer Mimi codes
        self.code_embedding = nn.Embedding(
            config.codebook_size * config.num_quantizers,
            config.hidden_size,
        )

        # Pre-transformer for temporal context
        self.pre_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Upsampling blocks
        # Upsampling from 12.5 Hz to 24 kHz requires ~1920x upsampling
        self.upsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(
                        config.hidden_size,
                        config.hidden_size,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                )
                for _ in range(10)  # 2^10 = 1024x, need more for 1920x
            ]
        )

        # Final decoder to waveform
        self.decoder = nn.Conv1d(
            config.hidden_size,
            1,
            kernel_size=3,
            padding=1,
        )

        self.make_empty_intermediate_tensors = lambda: None

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """
        Convert 8-layer Mimi codes to audio waveform.

        Args:
            input_ids: [batch, seq_len] Mimi code IDs (flattened 8 layers)

        Returns:
            [batch, 1, waveform_len] audio waveform at 24 kHz
        """
        assert input_ids is not None, "input_ids must be provided"

        # Embed codes
        x = self.code_embedding(input_ids)  # [batch, seq_len, hidden_size]

        # Pre-transformer for temporal context
        x = self.pre_transformer(x)  # [batch, seq_len, hidden_size]

        # Transpose for conv1d
        x = x.transpose(1, 2)  # [batch, hidden_size, seq_len]

        # Upsample
        for upsample_block in self.upsample:
            x = upsample_block(x)

        # Decode to waveform
        waveform = self.decoder(x)  # [batch, 1, waveform_len]

        return waveform

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            pass
        return loaded
