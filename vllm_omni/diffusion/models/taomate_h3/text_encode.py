# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph-safe text-only prompt encoding for TaoMate-H3 prompt updates.

A prompt update re-encodes the prompt on the workers with the MiniMax-H3
Qwen3-VL text encoder. For a few dozen tokens the 50 retained decoder layers
are launch-bound (about 0.15-0.27 s per prompt, measured locally, for a few
milliseconds of device work), and the encode runs inside the streaming step
loop, so every update costs the stream that much wall time.

``GraphSafeTextEncode`` supplies text-only positions and precomputed float32
rotary frequencies to the encoder's ordinary text-model forward. It shares
the embedding, rotary and decoder math with eager encoding and is captured
by ``GraphedForward`` per token count.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def cudnn_sdp_enabled(enabled: bool = True) -> Iterator[None]:
    """Mirror ``MiniMaxH3Qwen3VLEncoder.encode_ids``, which selects cuDNN SDPA for the encode."""
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(previous)


def text_only_positions(ids: torch.Tensor) -> torch.Tensor:
    """mRoPE positions [3, 1, L] of a prompt without image or video tokens."""
    length = int(ids.shape[-1])
    return torch.arange(length, device=ids.device, dtype=ids.dtype).view(1, 1, -1).expand(3, 1, -1)


class GraphSafeTextEncode(nn.Module):
    """``encoder._encode`` for text-only prompts, written for CUDA-graph capture."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        rotary = getattr(encoder.text_model, "rotary_emb", None)
        get_inv_freq = getattr(rotary, "_get_inv_freq", None)
        device = getattr(encoder, "device_target", None)
        self.inv_freq: torch.Tensor | None = None
        if callable(get_inv_freq) and device is not None:
            # One host-to-device copy at construction instead of one per forward.
            self.inv_freq = get_inv_freq(torch.device(device)).contiguous()

    def forward(self, *, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        text_model = self.encoder.text_model
        ids = input_ids.view(1, -1).to(torch.long)
        hidden = text_model.embed_tokens(ids)
        positions = text_only_positions(ids)
        if self.inv_freq is None:
            hidden = text_model(hidden, positions)
        else:
            hidden = text_model(hidden, positions, rotary_inv_freq=self.inv_freq)
        return hidden[0].to(torch.bfloat16)


__all__ = [
    "GraphSafeTextEncode",
    "cudnn_sdp_enabled",
    "text_only_positions",
]
