# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Conditioning encoder and tokenization utilities for Magenta RealTime 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.transformers_utils.configs.magenta_rt2 import MagentaRT2Config

NUM_RESERVED_TOKENS = 6
TOKEN_OFFSET = NUM_RESERVED_TOKENS + 1  # +1 shifts masked (-1) -> 6, valid 0.. -> 7..

NUM_MULAN_CHANNELS = 12
MULAN_CODEBOOK_SIZE = 1024
MULAN_VOCAB_PER_CHANNEL = 1031  # 1024 + 6 reserved + 1 dropout

NUM_PIANOROLL_CHANNELS = 128
PIANOROLL_VOCAB_PER_CHANNEL = 11  # 4 + 6 reserved + 1 dropout

NUM_DRUM_CHANNELS = 1
DRUM_VOCAB_PER_CHANNEL = 9  # 2 + 6 reserved + 1 dropout

NUM_CFG_MUSICCOCA_NOTES_CHANNELS = 2
CFG_MUSICCOCA_NOTES_VOCAB = 47  # 41 + 6 reserved
CFG_MUSICCOCA_NOTES_STEP = 0.2
CFG_MUSICCOCA_NOTES_MAX_BIN = 40

NUM_CFG_DRUMS_CHANNELS = 1
CFG_DRUMS_VOCAB = 15  # 9 + 6 reserved
CFG_DRUMS_STEP = 1.0
CFG_DRUMS_MAX_BIN = 8

NUM_REGULAR_CHANNELS = (
    NUM_PIANOROLL_CHANNELS + NUM_DRUM_CHANNELS + NUM_CFG_MUSICCOCA_NOTES_CHANNELS + NUM_CFG_DRUMS_CHANNELS
)  # 132
ENCODER_NUM_CHANNELS = NUM_MULAN_CHANNELS + NUM_REGULAR_CHANNELS  # 144

DEFAULT_CFG_SCALES: dict[str, float] = {
    "musiccoca": 3.0,
    "notes": 1.0,
    "drums": 1.0,
}


def regular_channel_vocab_sizes() -> tuple[int, ...]:
    """Per-channel vocabulary sizes for the 132 non-MuLan encoder channels."""
    return (
        (PIANOROLL_VOCAB_PER_CHANNEL,) * NUM_PIANOROLL_CHANNELS
        + (DRUM_VOCAB_PER_CHANNEL,) * NUM_DRUM_CHANNELS
        + (CFG_MUSICCOCA_NOTES_VOCAB,) * NUM_CFG_MUSICCOCA_NOTES_CHANNELS
        + (CFG_DRUMS_VOCAB,) * NUM_CFG_DRUMS_CHANNELS
    )


def discretize_cfg(value: float, step: float, max_bin: int) -> int:
    """Map a CFG scale in ``[-1.0, 7.0]`` to a discrete conditioning token index."""
    clamped = max(-1.0, min(7.0, float(value)))
    bin_index = round((clamped - (-1.0)) / step)
    return max(0, min(max_bin, int(bin_index)))


def _coerce_channel_tokens(
    value: Any,
    expected_len: int,
    *,
    default_val: int = -1,
) -> list[int]:
    if value is None:
        return [default_val] * expected_len
    if isinstance(value, np.ndarray):
        arr = [int(v) for v in value.reshape(-1).tolist()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        arr = [int(v) for v in value]
    else:
        raise ValueError(
            f"Expected sequence of {expected_len} ints for conditioning channel, got {type(value).__name__}"
        )
    if len(arr) != expected_len:
        raise ValueError(f"Expected {expected_len} tokens, got {len(arr)}")
    return arr


def build_conditioning_tokens(
    *,
    mulan_tokens: Sequence[int] | np.ndarray | None = None,
    pianoroll_tokens: Sequence[int] | np.ndarray | None = None,
    drum_tokens: Sequence[int] | np.ndarray | None = None,
    cfg_scales: Mapping[str, float] | None = None,
    raw_conditioning_tokens: Sequence[int] | np.ndarray | None = None,
    already_offset: bool = False,
) -> np.ndarray:
    """Build the 144-int encoder input frame matching ``MagentaRT2System._build_conditioning``."""
    if raw_conditioning_tokens is not None:
        arr = np.asarray(raw_conditioning_tokens, dtype=np.int32).reshape(-1)
        if arr.shape[0] != ENCODER_NUM_CHANNELS:
            raise ValueError(f"Expected raw conditioning vector of length {ENCODER_NUM_CHANNELS}, got {arr.shape[0]}")
        return arr.copy() if already_offset else (arr + TOKEN_OFFSET)

    scales = dict(DEFAULT_CFG_SCALES)
    if cfg_scales:
        for k, v in cfg_scales.items():
            if v is not None:
                scales[str(k)] = float(v)

    mulan = _coerce_channel_tokens(mulan_tokens, NUM_MULAN_CHANNELS, default_val=-1)
    pianoroll = _coerce_channel_tokens(pianoroll_tokens, NUM_PIANOROLL_CHANNELS, default_val=-1)
    drums = _coerce_channel_tokens(drum_tokens, NUM_DRUM_CHANNELS, default_val=-1)
    cfg_musiccoca_notes = [
        discretize_cfg(scales["musiccoca"], CFG_MUSICCOCA_NOTES_STEP, CFG_MUSICCOCA_NOTES_MAX_BIN),
        discretize_cfg(scales["notes"], CFG_MUSICCOCA_NOTES_STEP, CFG_MUSICCOCA_NOTES_MAX_BIN),
    ]
    cfg_drums = [
        discretize_cfg(scales["drums"], CFG_DRUMS_STEP, CFG_DRUMS_MAX_BIN),
    ]
    raw = np.asarray(
        mulan + pianoroll + drums + cfg_musiccoca_notes + cfg_drums,
        dtype=np.int32,
    )
    return raw if already_offset else (raw + TOKEN_OFFSET)


class MagentaRT2ConditioningEncoder(nn.Module):
    """Branched conditioning encoder (MuLan RVQ sum + regular MultiChannel mean -> LayerNorm)."""

    def __init__(self, config: MagentaRT2Config) -> None:
        super().__init__()
        self.config = config
        self.num_mulan_channels = int(config.num_mulan_channels)
        self.mulan_vocab_per_channel = int(config.mulan_vocab_per_channel)
        self.num_regular_channels = int(config.num_regular_channels)
        self.regular_vocab_size = int(config.regular_vocab_size)
        self.encoder_dim = int(config.encoder_dim)
        self.norm_eps = float(config.norm_eps)

        self.mulan_dequantizer = nn.Embedding(
            self.num_mulan_channels * self.mulan_vocab_per_channel,
            int(config.mulan_dim),
        )
        self.depth_input_adapter = nn.Linear(
            int(config.mulan_dim),
            self.encoder_dim,
            bias=False,
        )
        self.regular_embedder = nn.Embedding(
            self.regular_vocab_size,
            self.encoder_dim,
        )
        self.encoder_ln = nn.LayerNorm(self.encoder_dim, eps=self.norm_eps)

        mulan_offsets = torch.arange(self.num_mulan_channels, dtype=torch.long) * self.mulan_vocab_per_channel
        self.register_buffer("mulan_offsets", mulan_offsets, persistent=False)

        vocab_sizes = regular_channel_vocab_sizes()
        if len(vocab_sizes) != self.num_regular_channels:
            raise ValueError(f"Expected {self.num_regular_channels} regular channels, got {len(vocab_sizes)}")
        reg_offsets = [0]
        for sz in vocab_sizes[:-1]:
            reg_offsets.append(reg_offsets[-1] + sz)
        self.register_buffer(
            "regular_offsets",
            torch.tensor(reg_offsets, dtype=torch.long),
            persistent=False,
        )

    def forward(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, ..., 144]`` conditioning tokens into ``[B, ..., encoder_dim]``."""
        cond_long = cond_tokens.to(dtype=torch.long)
        mulan_ids = cond_long[..., : self.num_mulan_channels]
        mulan_ids = mulan_ids.clamp(0, self.mulan_vocab_per_channel - 1) + self.mulan_offsets
        # Sum across the 12 MuLan RVQ levels in fp32 for numeric parity with JAX/XLA.
        mulan_sum = (
            self.mulan_dequantizer(mulan_ids).float().sum(dim=-2).to(dtype=self.depth_input_adapter.weight.dtype)
        )
        mulan_emb = self.depth_input_adapter(mulan_sum)

        reg_ids = cond_long[..., self.num_mulan_channels :]
        reg_idx = (reg_ids + self.regular_offsets).clamp(0, self.regular_vocab_size - 1)
        reg_emb = self.regular_embedder(reg_idx).float().mean(dim=-2).to(dtype=mulan_emb.dtype)

        combined = 0.5 * (mulan_emb + reg_emb)
        # Apply encoder LayerNorm in float32 (matching reductions_in_at_least_fp32=True).
        return F.layer_norm(
            combined.float(),
            (self.encoder_dim,),
            self.encoder_ln.weight.float(),
            self.encoder_ln.bias.float(),
            self.norm_eps,
        ).to(dtype=combined.dtype)


__all__ = [
    "DEFAULT_CFG_SCALES",
    "ENCODER_NUM_CHANNELS",
    "NUM_MULAN_CHANNELS",
    "NUM_REGULAR_CHANNELS",
    "NUM_RESERVED_TOKENS",
    "TOKEN_OFFSET",
    "MagentaRT2ConditioningEncoder",
    "build_conditioning_tokens",
    "discretize_cfg",
    "regular_channel_vocab_sizes",
]
