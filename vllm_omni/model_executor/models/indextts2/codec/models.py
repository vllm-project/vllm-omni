# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the Amphion project
"""Inference-only EnhancedCodec decoder for IndexTTS 2.5.

The checkpoint originates from the Amphion EnhancedCodec implementation
vendored by IndexTTS 2.5. The encoder and training paths are deliberately not
constructed because vLLM-Omni only performs semantic-code decoding here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import ResidualVQDecoder
from .vocos import VocosBackbone

logger = logging.getLogger(__name__)


def _config_value(
    cfg: Mapping[str, Any] | Any | None,
    name: str,
    default: Any,
) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


class EnhancedCodec(nn.Module):
    """Subset of EnhancedCodec required for ``decode(codes)``."""

    def __init__(
        self,
        codebook_size: int = 8192,
        hidden_size: int = 1024,
        codebook_dim: int = 8,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 2048,
        vocos_num_layers: int = 12,
        num_quantizers: int = 1,
        downsample_scale: int | None = 2,
        cfg: Mapping[str, Any] | Any | None = None,
    ):
        super().__init__()
        codebook_size = int(_config_value(cfg, "codebook_size", codebook_size))
        hidden_size = int(_config_value(cfg, "hidden_size", hidden_size))
        codebook_dim = int(_config_value(cfg, "codebook_dim", codebook_dim))
        vocos_dim = int(_config_value(cfg, "vocos_dim", vocos_dim))
        vocos_intermediate_dim = int(_config_value(cfg, "vocos_intermediate_dim", vocos_intermediate_dim))
        vocos_num_layers = int(_config_value(cfg, "vocos_num_layers", vocos_num_layers))
        num_quantizers = int(_config_value(cfg, "num_quantizers", num_quantizers))
        downsample_scale = _config_value(
            cfg,
            "downsample_scale",
            downsample_scale,
        )

        self.hidden_size = hidden_size
        self.num_quantizers = num_quantizers
        self.downsample_scale = downsample_scale
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.quantizer = ResidualVQDecoder(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )
        if downsample_scale is not None and downsample_scale > 1:
            self.up = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode `[B,T]` or `[N,B,T]` codes to `[B,T*scale,hidden]`."""
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        if codes.ndim != 3:
            raise ValueError(f"EnhancedCodec codes must have shape [B,T] or [N,B,T], got {tuple(codes.shape)}")
        if codes.shape[0] != self.num_quantizers:
            raise ValueError(f"EnhancedCodec expected {self.num_quantizers} codebooks, got {codes.shape[0]}")

        hidden = self.quantizer.vq2emb(codes)
        hidden = self.decoder(hidden)
        if self.downsample_scale is not None and self.downsample_scale > 1:
            hidden = hidden.transpose(1, 2)
            hidden = F.interpolate(
                hidden,
                scale_factor=self.downsample_scale,
                mode="nearest",
            )
            hidden = self.up(hidden).transpose(1, 2)
        return hidden

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"EnhancedCodec checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved_state = checkpoint.get("model", checkpoint)
        own_state = self.state_dict()
        missing = [
            name
            for name, value in own_state.items()
            if name not in saved_state or saved_state[name].shape != value.shape
        ]
        if missing:
            raise RuntimeError("EnhancedCodec checkpoint is missing required decoder weights: " + ", ".join(missing))
        self.load_state_dict(
            {name: saved_state[name] for name in own_state},
            strict=True,
        )
        logger.info(
            "Loaded %d inference-only EnhancedCodec tensors from %s",
            len(own_state),
            path,
        )
