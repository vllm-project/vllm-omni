# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Ollin Boer Bohan
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiny approximate decoder (TAEH3) for MiniMax-H3 video latents."""

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

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

TAEH3_LATENT_CHANNELS = 24
# The H3 video VAE works in 17-frame clips of five latent frames. Each clip
# decodes to 5 * 4 = 20 frames whose first three are padding, and the last
# three latent frames of a video are dropped at encode time. A video of n clips
# is therefore 5n - 3 latent frames and 17n - 12 pixel frames.
_CLIP_LATENT_FRAMES = 5
_FRAMES_PER_LATENT_FRAME = 4
_CLIP_PADDING_FRAMES = 3
_DROPPED_LATENT_FRAMES = 3
_CLIP_FRAMES = _CLIP_LATENT_FRAMES * _FRAMES_PER_LATENT_FRAME - _CLIP_PADDING_FRAMES
_SPATIAL_SCALE = 16
_PATCH_SIZE = 2

TAEH3ChunkCallback = Callable[[torch.Tensor], None]


def resolve_h3_tae_decoder_path(additional: Mapping[str, object] | None) -> str | None:
    """Read ``tae_decoder_path`` from ``additional_config``, rejecting conflicting decoders."""
    if not additional or "tae_decoder_path" not in additional:
        return None
    value = additional["tae_decoder_path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("tae_decoder_path must be a nonempty local file path")
    for other in ("video_vae_decoder", "lightvae_decoder_path"):
        if additional.get(other) is not None:
            raise ValueError(f"tae_decoder_path cannot be combined with {other}")
    return value


def taeh3_decoded_frames(latent_frames: int) -> int:
    """Pixel frames the H3 video VAE produces for ``latent_frames`` latent frames."""
    clips, remainder = divmod(int(latent_frames) + _DROPPED_LATENT_FRAMES, _CLIP_LATENT_FRAMES)
    if remainder or clips < 1:
        raise ValueError(
            "TAEH3 decodes MiniMax-H3 videos of 5n - 3 latent frames (2, 7, 12, ...); "
            f"got {int(latent_frames)} latent frames"
        )
    return clips * _CLIP_FRAMES - _DROPPED_LATENT_FRAMES * _FRAMES_PER_LATENT_FRAME


def _conv(in_channels: int, out_channels: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3) * 3


class _MemBlock(nn.Module):
    """Residual block that also sees the previous frame's input to the block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(channels * 2, channels),
            nn.ReLU(inplace=True),
            _conv(channels, channels),
            nn.ReLU(inplace=True),
            _conv(channels, channels),
        )
        self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([value, past], dim=1)) + self.skip(value))


class _TGrow(nn.Module):
    """Turn every frame into ``stride`` consecutive frames."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        frames, channels, height, width = value.shape
        return self.conv(value).reshape(frames * self.stride, channels, height, width)


class TAEH3Decoder(nn.Module):
    """Decode MiniMax-H3 video latents with the ``taeh3`` tiny autoencoder.

    The decoder takes the latents the DiT produces, without the VAE's latent
    mean and standard deviation applied, and returns frames in ``[0, 1]``. It
    is an approximation of the H3 video VAE's decoder with about 250 times
    fewer parameters, meant for previews and for hosts that cannot hold the
    full decoder. It does not reproduce the full decoder's pixels.

    A video is decoded one clip (five latent frames) at a time. The only state
    carried between clips is the last input of each memory block, so device
    memory does not grow with the video length.
    """

    def __init__(self, checkpoint_path: str | Path, *, device: torch.device, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        channels = (256, 128, 64, 64)
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(TAEH3_LATENT_CHANNELS, channels[0]),
            nn.ReLU(inplace=True),
            _MemBlock(channels[0]),
            _MemBlock(channels[0]),
            _MemBlock(channels[0]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[0], 1),
            _conv(channels[0], channels[1], bias=False),
            _MemBlock(channels[1]),
            _MemBlock(channels[1]),
            _MemBlock(channels[1]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[1], 2),
            _conv(channels[1], channels[2], bias=False),
            _MemBlock(channels[2]),
            _MemBlock(channels[2]),
            _MemBlock(channels[2]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[2], 2),
            _conv(channels[2], channels[3], bias=False),
            nn.ReLU(inplace=True),
            _conv(channels[3], 3 * _PATCH_SIZE**2),
        )
        self.decoder.load_state_dict(_read_decoder_state(checkpoint_path), strict=True)
        if dtype is None:
            # The published weights are FP16. Host convolutions want FP32.
            dtype = torch.float32 if device.type == "cpu" else torch.float16
        self.to(device=device, dtype=dtype).eval().requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder[1].weight.dtype

    @property
    def device(self) -> torch.device:
        return self.decoder[1].weight.device

    def _decode_clip(self, latent: torch.Tensor, memory: list[torch.Tensor | None]) -> torch.Tensor:
        """Decode ``[B, T, C, h, w]`` latent frames to ``[B, 4T, 3, 16h, 16w]`` frames."""
        batch = latent.shape[0]
        value = latent.flatten(0, 1)
        for index, block in enumerate(self.decoder):
            if isinstance(block, _MemBlock):
                frames = value.unflatten(0, (batch, -1))
                first = memory[index]
                if first is None:
                    first = torch.zeros_like(frames[:, 0])
                past = torch.cat([first.unsqueeze(1), frames[:, :-1]], dim=1).flatten(0, 1)
                memory[index] = frames[:, -1].clone()
                value = block(value, past)
            else:
                value = block(value)
        return F.pixel_shuffle(value, _PATCH_SIZE).clamp_(0, 1).unflatten(0, (batch, -1))

    def iter_clips(self, latent: torch.Tensor) -> Iterator[torch.Tensor]:
        """Yield the video clip by clip as ``[B, 3, T, H, W]`` frames in ``[0, 1]``.

        Every clip has 17 frames except the last, which has five.
        """
        _check_latent(latent)
        latent = latent.to(device=self.device, dtype=self.dtype).transpose(1, 2)
        memory: list[torch.Tensor | None] = [None] * len(self.decoder)
        for start in range(0, latent.shape[1], _CLIP_LATENT_FRAMES):
            frames = self._decode_clip(latent[:, start : start + _CLIP_LATENT_FRAMES], memory)
            yield frames[:, _CLIP_PADDING_FRAMES:].transpose(1, 2)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, 24, T, h, w]`` latents to ``[B, 3, F, 16h, 16w]`` float frames in ``[0, 1]``."""
        frames = _check_latent(latent)
        batch, _, _, height, width = latent.shape
        # Fill one output tensor in place instead of holding every clip twice.
        video = torch.empty(
            (batch, 3, frames, height * _SPATIAL_SCALE, width * _SPATIAL_SCALE),
            dtype=torch.float32,
            device=self.device,
        )
        offset = 0
        for clip in self.iter_clips(latent):
            video[:, :, offset : offset + clip.shape[2]] = clip
            offset += clip.shape[2]
        return video


def _check_latent(latent: torch.Tensor) -> int:
    """Validate a latent video and return the number of frames it decodes to."""
    if latent.ndim != 5 or latent.shape[1] != TAEH3_LATENT_CHANNELS:
        raise ValueError(
            f"TAEH3 expects [B, {TAEH3_LATENT_CHANNELS}, T, H, W] latents, got shape {tuple(latent.shape)}"
        )
    return taeh3_decoded_frames(latent.shape[2])


def _read_decoder_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    if path.suffix != ".safetensors" or not path.is_file():
        raise ValueError(f"tae_decoder_path must name a local .safetensors file, got {str(checkpoint_path)!r}")
    prefix = "decoder."
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        # The published file also carries the matching encoder, which is not used.
        state = {
            name[len(prefix) :]: checkpoint.get_tensor(name) for name in checkpoint.keys() if name.startswith(prefix)
        }
    if not state:
        raise ValueError(f"{path} holds no TAEH3 decoder weights (no 'decoder.*' tensors)")
    latent_weight = state.get("1.weight")
    if latent_weight is None or latent_weight.ndim != 4 or latent_weight.shape[1] != TAEH3_LATENT_CHANNELS:
        raise ValueError(
            f"{path} is not a TAEH3 checkpoint: its decoder does not take {TAEH3_LATENT_CHANNELS}-channel latents"
        )
    return state


def decode_taeh3_chunks(
    decoder: TAEH3Decoder,
    latent: torch.Tensor,
    callback: TAEH3ChunkCallback,
    *,
    group: dist.ProcessGroup | None,
) -> None:
    """Publish the video clip by clip from the rank that owns the output.

    The tiny decoder runs on one device and uses no collectives, so peer ranks
    of a distributed VAE group decode nothing. They still learn whether the
    owner's callback failed, so that every rank raises or none does.
    """
    owner = group is None or dist.get_rank(group) == 0
    error: BaseException | None = None
    if owner:
        for clip in decoder.iter_clips(latent):
            try:
                callback(clip.float().contiguous())
            except BaseException as exc:  # noqa: BLE001
                error = exc
                break
    if group is not None:
        failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=latent.device)
        dist.broadcast(failed, src=dist.get_global_rank(group, 0), group=group)
        if int(failed.item()) and error is None:
            raise RuntimeError("MiniMax-H3 video chunk callback failed on rank zero")
    if error is not None:
        raise error


__all__ = [
    "TAEH3_LATENT_CHANNELS",
    "TAEH3Decoder",
    "decode_taeh3_chunks",
    "resolve_h3_tae_decoder_path",
    "taeh3_decoded_frames",
]
