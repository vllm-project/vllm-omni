"""Vendored TADA codec decoder.

Adapted from HumeAI's ``hume-tada`` package (``tada/modules/decoder.py``), which
is MIT-licensed. Vendored so the vLLM-Omni TADA integration does not depend on the
``hume-tada`` package at runtime; codec *weights* are still loaded from
``HumeAI/tada-codec`` via ``from_pretrained(subfolder="decoder")``. Class and
parameter names are kept identical so the checkpoint loads without remapping.

SPDX-License-Identifier: MIT
Copyright (c) Hume AI.
"""

import math
from typing import Literal

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from .layers import LocalAttentionEncoder, ResidualUnit, Snake1d, WNConv1d, WNConvTranspose1d


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class DACDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def _create_segment_attention_mask(
    text_token_mask: torch.Tensor, version: Literal["v1", "v2", "decoder_block_attention"] = "v1"
) -> torch.Tensor:
    """Decoder-side block attention mask.

    NOTE: this is the *decoder's* variant and differs from the encoder's mask of
    the same name — here block ids use ``cumsum - mask`` (marked positions belong
    to the preceding block) and v2 simply allows same-block + previous-block. Kept
    verbatim from upstream; the trained decoder weights assume exactly this rule.
    Returns a boolean mask where True means *masked* (cannot attend).
    """
    if version == "v1":
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_i == block_ids_j

        _, seq_len = text_token_mask.shape
        positions = torch.arange(seq_len, device=text_token_mask.device)
        pos_i = positions.unsqueeze(1)
        pos_j = positions.unsqueeze(0)

        is_marked_i = text_token_mask.unsqueeze(2).bool()
        marked_causal = (pos_j <= pos_i).unsqueeze(0) & is_marked_i
        past = (pos_j < pos_i).unsqueeze(0)
        current_block_forward = (pos_j >= pos_i) & same_block
        non_marked_attention = (past | current_block_forward) & ~is_marked_i
        can_attend = marked_causal | non_marked_attention
        return ~can_attend
    elif version == "v2":
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_j == block_ids_i
        prev_block = block_ids_j == (block_ids_i - 1)
        can_attend = same_block | prev_block
        return ~can_attend
    else:
        raise ValueError(f"Unknown version: {version}")


class DecoderConfig(PretrainedConfig):
    # Defined via __init__ (not class attributes) so newer transformers, which
    # treats PretrainedConfig subclasses as dataclasses, does not reject the
    # mutable ``strides`` default.
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_attn_layers: int = 6,
        num_attn_heads: int = 8,
        attn_dim_feedforward: int = 4096,
        attn_dropout: float = 0.1,
        use_flash_attn: bool = True,
        wav_decoder_channels: int = 1536,
        strides: list[int] | None = None,
        block_attention: Literal["none", "v1", "v2"] = "v2",
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_attn_layers = num_attn_layers
        self.num_attn_heads = num_attn_heads
        self.attn_dim_feedforward = attn_dim_feedforward
        self.attn_dropout = attn_dropout
        self.use_flash_attn = use_flash_attn
        self.wav_decoder_channels = wav_decoder_channels
        self.strides = [4, 4, 5, 6] if strides is None else strides
        self.block_attention = block_attention
        super().__init__(**kwargs)


class Decoder(PreTrainedModel):
    config_class = DecoderConfig

    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.decoder_proj = nn.Linear(self.config.embed_dim, self.config.hidden_dim)

        self.local_attention_decoder = LocalAttentionEncoder(
            d_model=self.config.hidden_dim,
            num_layers=self.config.num_attn_layers,
            num_heads=self.config.num_attn_heads,
            d_ff=self.config.attn_dim_feedforward,
            dropout=self.config.attn_dropout,
            activation="gelu",
            max_seq_len=8192,
            use_flash_attn=self.config.use_flash_attn,
        )
        self.wav_decoder = DACDecoder(
            input_channel=self.config.hidden_dim,
            channels=self.config.wav_decoder_channels,
            rates=self.config.strides,
        )
        # Required by newer transformers (initializes tied-weights bookkeeping
        # used during from_pretrained). Upstream hume-tada omits it because it
        # predates that requirement.
        self.post_init()

    def forward(self, encoded_expanded: torch.Tensor, token_masks: torch.Tensor):
        decoder_input = self.decoder_proj(encoded_expanded)
        # Apply decoder block attention if text_token_mask is provided
        attn_mask = _create_segment_attention_mask(token_masks, version="v2")
        decoded_expanded = self.local_attention_decoder(decoder_input, mask=attn_mask)

        x_rec = self.wav_decoder(decoded_expanded.transpose(1, 2))
        return x_rec

    def generate(self, encoded_expanded: torch.Tensor, **kwargs):
        return self.forward(encoded_expanded, **kwargs)
