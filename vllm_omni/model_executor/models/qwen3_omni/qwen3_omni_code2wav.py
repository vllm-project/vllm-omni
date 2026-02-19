# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe Code2Wav model."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
    SnakeBeta,
)
from vllm.config import VllmConfig  # type: ignore
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger  # type: ignore
from vllm.model_executor.models.utils import (  # type: ignore
    AutoWeightsLoader,
    WeightsMapper,
)

logger = init_logger(__name__)


class Qwen3OmniMoeCode2Wav(nn.Module):
    """
    Qwen3 Omni MoE Code2Wav - Converts num_quantizers-layer RVQ codec codes to audio waveform.

    Architecture:
    1. Code Embedding: Embed and average num_quantizers RVQ layers
    2. Pre-Transformer: Add temporal context via sliding-window attention
    3. Upsampling: Progressive upsampling with ConvNeXt blocks
    4. Decoder: Multi-stage upsampling + residual units → waveform

    Input: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
    Output: [batch, 1, waveform_len] - Audio waveform [-1, 1]

    Total upsampling factor: ~1280x
    Example: 100 codec frames → 128,000 audio samples (8 seconds at 16kHz)
    """

    input_modalities = "audio"

    # Weight mapper
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "code2wav.pre_transformer.": "pre_transformer.",
            "code2wav.code_embedding.": "code_embedding.",
            "code2wav.upsample.": "upsample.",
            "code2wav.decoder.": "decoder.",
            "code2wav.": "",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config: Qwen3OmniMoeCode2WavConfig = vllm_config.model_config.hf_config

        # Calculate total upsampling factor
        self.total_upsample = np.prod(self.config.upsample_rates + self.config.upsampling_ratios)

        # Pre-transformer
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(self.config)

        # Code embedding: Single embedding table for all RVQ layers
        self.code_embedding = nn.Embedding(
            self.config.codebook_size * self.config.num_quantizers, self.config.hidden_size
        )

        # Offset for each RVQ layer (layer 0: 0-1023, layer 1: 1024-2047, etc.)
        self.register_buffer(
            "code_offset",
            torch.arange(self.config.num_quantizers).view(1, -1, 1) * self.config.codebook_size,
            persistent=False,
        )

        # Upsampling blocks (e.g., 2x, 2x)
        upsample = []
        for factor in self.config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(
                            self.config.hidden_size, self.config.hidden_size, factor, factor
                        ),
                        Qwen3OmniMoeConvNeXtBlock(self.config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        # Decoder: Initial projection + progressive upsampling blocks
        decoder = [Qwen3OmniMoeCausalConvNet(self.config.hidden_size, self.config.decoder_dim, kernel_size=7)]

        # Add decoder blocks (each upsamples and reduces channels)
        for i in range(len(self.config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(self.config, i))

        # Final projection to waveform
        output_dim = self.config.decoder_dim // 2 ** len(self.config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, kernel_size=7),
        ]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert num_quantizers-layer RVQ codes to audio waveform.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codec codes

        Returns:
            waveform: [batch, 1, waveform_len] - Audio waveform clipped to [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layers of codes, got {codes.shape[1]}")

        # Stage 1: Code Embedding
        # Add offset to separate layer vocabularies, then embed and average
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        # Shape: [batch, seq_len, hidden_size]

        # Stage 2: Pre-Transformer (add temporal context)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        # Shape: [batch, seq_len, hidden_size]

        # Stage 3: Upsampling
        hidden = hidden.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        # Shape: [batch, hidden_size, seq_len * upsample_factor]

        # Stage 4: Decoder (progressive upsampling to waveform)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        # Shape: [batch, 1, waveform_len]

        # Clamp to valid audio range
        return wav.clamp(min=-1.0, max=1.0)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            chunk_size: Number of codec frames per chunk
            left_context_size: Number of overlapping frames for context

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        wavs = []
        start_index = 0

        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index >= left_context_size else start_index

            # Extract chunk with left context
            codes_chunk = codes[..., start_index - context_size : end_index]

            # Decode chunk
            wav_chunk = self(codes_chunk)

            # Remove context from output (context_size * total_upsample samples)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])

            start_index = end_index

        ubatch_slices = get_forward_context().ubatch_slices
        if ubatch_slices is not None:
            code_seq_lens = [seq_len // self.config.num_quantizers for seq_len in ubatch_slices]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            # Create one entry per batch so that each element is processed.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        batch_wav = torch.cat(wavs, dim=-1)
        wavs = []
        for idx, code_seq_len in enumerate(code_seq_lens):
            wav_chunk = batch_wav[idx, :, : code_seq_len * self.total_upsample]
            wavs.append(wav_chunk)
        return wavs

    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        chunk_size: int = 25,
        left_context_size: int = 25,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            chunk_size: Number of codec frames per chunk
            left_context_size: Number of overlapping frames for context

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        # Decode chunk
        wavs = []
        batch_wav = self(codes)
        ubatch_slices = get_forward_context().ubatch_slices
        if ubatch_slices is not None:
            code_seq_lens = [seq_len // self.config.num_quantizers for seq_len in ubatch_slices]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            # Create one entry per batch so that each element is processed.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        for idx, code_seq_len in enumerate(code_seq_lens):
            # TODO: need to optimize algorithms, current only support
            # chunk_size = left_context_size = 25
            if code_seq_len <= chunk_size:
                context_size = 0
            else:
                context_size = left_context_size
            # Remove context from output (context_size * total_upsample samples)
            wav_chunk = batch_wav[idx, :, context_size * self.total_upsample : code_seq_len * self.total_upsample]
            wavs.append(wav_chunk)
        return wavs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "talker."],  # Already loaded above
        )
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
            logger.error("Error logging model load summary")

        return loaded
