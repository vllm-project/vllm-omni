# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Ollin Boer Bohan
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight stateful TAEW2.2 decoder used by ABot-World."""

# Decoder architecture adapted from https://github.com/madebyollin/taehv.
# The original portions carry the following MIT license:
# Copyright (c) 2025 Ollin Boer Bohan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.experimental.ar_diffusion.streaming_decode import StreamingDecodeState


def _conv(in_channels: int, out_channels: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3) * 3


class _MemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(in_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([value, past], dim=1)) + self.skip(value))


class _TGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        return self.conv(value).reshape(batch * self.stride, channels, height, width)


class TAEW2Decoder(nn.Module):
    """TAEW2.2 streaming decoder with explicit per-session MemBlock state."""

    @staticmethod
    def persistent_state_bytes(
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
    ) -> int:
        """Return the bytes retained by one streaming decoder session."""
        # Three MemBlocks retain one feature map at each 1x, 2x, and 4x stage.
        elements = 3 * latent_height * latent_width * (256 + 4 * 128 + 16 * 64)
        return elements * dtype.itemsize

    def __init__(self, checkpoint_path: str, dtype: torch.dtype) -> None:
        super().__init__()
        channels = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(48, channels[0]),
            nn.ReLU(inplace=True),
            _MemBlock(channels[0], channels[0]),
            _MemBlock(channels[0], channels[0]),
            _MemBlock(channels[0], channels[0]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[0], 1),
            _conv(channels[0], channels[1], bias=False),
            _MemBlock(channels[1], channels[1]),
            _MemBlock(channels[1], channels[1]),
            _MemBlock(channels[1], channels[1]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[1], 2),
            _conv(channels[1], channels[2], bias=False),
            _MemBlock(channels[2], channels[2]),
            _MemBlock(channels[2], channels[2]),
            _MemBlock(channels[2], channels[2]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[2], 2),
            _conv(channels[2], channels[3], bias=False),
            nn.ReLU(inplace=True),
            _conv(channels[3], 12),
        )
        source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        decoder_state = {
            name.removeprefix("decoder."): value for name, value in source.items() if name.startswith("decoder.")
        }
        self.decoder.load_state_dict(decoder_state, strict=True)
        self.to(dtype=dtype).eval().requires_grad_(False)
        self.dtype = dtype
        self.config = SimpleNamespace(
            z_dim=48,
            scale_factor_temporal=4,
            scale_factor_spatial=16,
            patch_size=2,
        )

    def new_decode_state(self, session_id: str) -> StreamingDecodeState:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return StreamingDecodeState(session_id=session_id, feat_map=[None] * len(self.decoder))

    def declared_state_bytes(self, *, height: int, width: int, dtype: torch.dtype) -> int:
        return self.persistent_state_bytes(height // 16, width // 16, dtype)

    def release(self, state: StreamingDecodeState) -> None:
        state.release()

    def _decode_one(self, latent: torch.Tensor, state: StreamingDecodeState) -> list[torch.Tensor]:
        queue: list[tuple[torch.Tensor, int]] = [(latent, 0)]
        output: list[torch.Tensor] = []
        while queue:
            value, index = queue.pop(0)
            if index == len(self.decoder):
                output.append(value)
                continue
            block = self.decoder[index]
            if isinstance(block, _MemBlock):
                past = state.feat_map[index]
                updated = block(value, torch.zeros_like(value) if past is None else past)
                state.feat_map[index] = value.clone()
                queue.insert(0, (updated, index + 1))
            elif isinstance(block, _TGrow):
                grown = block(value)
                for frame in reversed(grown.chunk(block.stride, dim=0)):
                    queue.insert(0, (frame, index + 1))
            else:
                queue.insert(0, (block(value), index + 1))
        return output

    @torch.inference_mode()
    def decode_chunk(
        self,
        latents: torch.Tensor,
        state: StreamingDecodeState,
    ) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != 48:
            raise ValueError("TAEW2.2 decode requires NCTHW latents with batch size 1 and 48 channels.")
        if latents.shape[2] == 0 or len(state.feat_map) != len(self.decoder):
            raise ValueError("TAEW2.2 requires non-empty latents and matching decoder state.")
        frames: list[torch.Tensor] = []
        for latent in latents.permute(0, 2, 1, 3, 4).unbind(dim=1):
            frames.extend(self._decode_one(latent, state))
        video = torch.stack(frames, dim=1)
        batch, time, channels, height, width = video.shape
        video = F.pixel_shuffle(video.reshape(batch * time, channels, height, width), 2)
        video = video.reshape(batch, time, 3, height * 2, width * 2).clamp_(0, 1)
        if not state.started:
            video = video[:, 3:]
        state.frames_decoded += latents.shape[2]
        state.chunks_decoded += 1
        return video.permute(0, 2, 1, 3, 4).mul(2).sub(1)
