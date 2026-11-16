# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/models/vibevoice_codec.py @ d3612b0.
# The codec implements the VibeVoice acoustic tokenizer architecture (Copyright (c) 2025 Microsoft, MIT License).
"""VibeVoice acoustic codec: 24 kHz audio <-> 64-dim latents at 7.5 Hz.

Both directions are causal convolution stacks. Every layer takes and returns
its left context explicitly, so the same code encodes a whole clip (zero
initial state) or one 3200-sample frame at a time with identical results.
States are plain tensors with a leading batch dimension and fixed shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.haru.configuration_haru import HaruCodecConfig

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.haru.codec_triton.stream import TritonCodecStream

State = list[torch.Tensor]


class RMSNorm(nn.Module):
    """RMS norm over the last dimension, computed in float32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).to(x.dtype)
        return y * self.weight


class CausalConv1d(nn.Module):
    """Conv1d over ``[B, C, T]`` whose left padding is an explicit state."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.in_channels, self.stride, self.groups = in_channels, stride, groups
        self.context = kernel - stride

    def state_shape(self) -> tuple[int, int]:
        return (self.in_channels, self.context)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tail = x.shape[-1] % self.stride
        if tail:  # final partial hop of a whole-clip encode
            x = F.pad(x, (0, self.stride - tail))
        x = torch.cat([state, x], dim=-1)
        y = F.conv1d(x, self.weight, self.bias, stride=self.stride, groups=self.groups)
        return y, x[..., x.shape[-1] - self.context :]


class CausalConvTranspose1d(nn.Module):
    """ConvTranspose1d with kernel = 2 * stride; the state is the overlap tail."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, 2 * stride))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.out_channels, self.stride = out_channels, stride

    def state_shape(self) -> tuple[int, int]:
        return (self.out_channels, self.stride)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = F.conv_transpose1d(x, self.weight, stride=self.stride)
        n = x.shape[-1] * self.stride
        head = y[..., :n]
        head = torch.cat([head[..., : self.stride] + state, head[..., self.stride :]], dim=-1)
        return head + self.bias[:, None], y[..., n:]


class Block(nn.Module):
    """Depthwise causal conv mixer and an MLP, each with a scaled residual."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.norm = RMSNorm(dim, eps)
        self.conv = CausalConv1d(dim, dim, 7, groups=dim)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.ffn_norm = RMSNorm(dim, eps)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.ffn_gamma = nn.Parameter(torch.ones(dim))

    def state_shape(self) -> tuple[int, int]:
        return self.conv.state_shape()

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, state = self.conv(self.norm(x.transpose(1, 2)).transpose(1, 2), state)
        x = x + y * self.gamma[:, None]
        h = self.fc2(F.gelu(self.fc1(self.ffn_norm(x.transpose(1, 2)))))
        return x + (h * self.ffn_gamma).transpose(1, 2), state


class _Stack(nn.Module):
    """A fixed sequence of stateful layers."""

    def sequence(self) -> list[nn.Module]:
        raise NotImplementedError

    def init_state(self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> State:
        ref = self.head.weight
        device, dtype = device or ref.device, dtype or ref.dtype
        return [torch.zeros(batch, *m.state_shape(), device=device, dtype=dtype) for m in self.sequence()]

    def run(self, x: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        new_state = []
        for module, s in zip(self.sequence(), state, strict=True):
            x, s = module(x, s)
            new_state.append(s)
        return x, new_state


class Encoder(_Stack):
    """``[B, 1, T]`` audio -> ``[B, T / 3200, latent]`` latent means."""

    def __init__(self, config: HaruCodecConfig):
        super().__init__()
        c = config.base_channels
        strides = config.ratios[::-1]
        self.stem = CausalConv1d(1, c, 7)
        self.down = nn.ModuleList(CausalConv1d(c << i, c << (i + 1), 2 * s, stride=s) for i, s in enumerate(strides))
        self.stages = nn.ModuleList(
            nn.ModuleList(Block(c << i, config.norm_eps) for _ in range(depth))
            for i, depth in enumerate(config.encoder_depths)
        )
        self.head = CausalConv1d(c << len(strides), config.latent_dim, 7)
        self._sequence: list[nn.Module] | None = None

    def sequence(self) -> list[nn.Module]:
        if self._sequence is None:
            seq: list[nn.Module] = []
            for i, stage in enumerate(self.stages):
                seq.append(self.stem if i == 0 else self.down[i - 1])
                seq.extend(stage)
            self._sequence = [*seq, self.head]
        return self._sequence

    def forward(self, audio: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        x, state = self.run(audio, state)
        return x.transpose(1, 2), state


class Decoder(_Stack):
    """``[B, T, latent]`` latents -> ``[B, T * 3200]`` audio."""

    def __init__(self, config: HaruCodecConfig):
        super().__init__()
        n = len(config.ratios)
        c = config.base_channels
        self.stem = CausalConv1d(config.latent_dim, c << n, 7)
        self.up = nn.ModuleList(
            CausalConvTranspose1d(c << (n - i), c << (n - i - 1), s) for i, s in enumerate(config.ratios)
        )
        self.stages = nn.ModuleList(
            nn.ModuleList(Block(c << (n - i), config.norm_eps) for _ in range(depth))
            for i, depth in enumerate(config.decoder_depths)
        )
        self.head = CausalConv1d(c, 1, 7)
        self._sequence: list[nn.Module] | None = None

    def sequence(self) -> list[nn.Module]:
        if self._sequence is None:
            seq: list[nn.Module] = []
            for i, stage in enumerate(self.stages):
                seq.append(self.stem if i == 0 else self.up[i - 1])
                seq.extend(stage)
            self._sequence = [*seq, self.head]
        return self._sequence

    def forward(self, latents: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        x, state = self.run(latents.transpose(1, 2), state)
        return x[:, 0], state


class VibeVoiceCodec(nn.Module):
    def __init__(self, config: HaruCodecConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.head.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.encoder.head.weight.device

    def sample(self, mean: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw latents around ``mean [B, T, D]`` (one noise scale per batch row)."""
        scale = torch.randn(mean.shape[0], 1, 1, device=mean.device, dtype=mean.dtype, generator=generator)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean + scale * (self.config.latent_std / 0.8) * noise

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Whole clip ``[T]`` or ``[B, T]`` -> latent means ``[B, ceil(T / 3200), D]``."""
        audio = audio.reshape(-1, audio.shape[-1]) if audio.dim() > 1 else audio[None]
        x = audio.to(self.device, self.dtype)[:, None]
        mean, _ = self.encoder(x, self.encoder.init_state(x.shape[0]))
        return mean

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Whole sequence ``[B, T, D]`` -> audio ``[B, T * 3200]``."""
        x = latents.to(self.device, self.dtype)
        audio, _ = self.decoder(x, self.decoder.init_state(x.shape[0]))
        return audio


class CodecStream:
    """Frame-by-frame encode and decode for ``batch`` parallel conversations."""

    def __init__(self, codec: VibeVoiceCodec, batch: int = 1):
        self.codec = codec
        self.batch = batch
        self.encoder_state = codec.encoder.init_state(batch)
        self.decoder_state = codec.decoder.init_state(batch)

    def reset(self, rows: torch.Tensor | None = None) -> None:
        """Clear all rows, or only the conversations at ``rows``."""
        for state in (*self.encoder_state, *self.decoder_state):
            if rows is None:
                state.zero_()
            else:
                state[rows] = 0

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """``[B, n * 3200]`` new audio -> latent means ``[B, n, D]``."""
        if audio.shape[-1] == 0 or audio.shape[-1] % self.codec.config.frame_samples:
            raise ValueError("streaming encode takes whole frames")
        x = audio.to(self.codec.device, self.codec.dtype)[:, None]
        mean, self.encoder_state = self.codec.encoder(x, self.encoder_state)
        return mean

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, n, D]`` new latents -> audio ``[B, n * 3200]``."""
        x = latents.to(self.codec.device, self.codec.dtype)
        audio, self.decoder_state = self.codec.decoder(x, self.decoder_state)
        return audio


def open_stream(codec: VibeVoiceCodec, batch: int = 1, backend: str = "torch") -> CodecStream | TritonCodecStream:
    """A frame-by-frame codec stream for ``batch`` conversations.

    ``backend``: ``"torch"`` (default), ``"triton"`` (Triton kernels replayed
    from CUDA graphs: about ten times faster per frame) or ``"auto"``, which
    picks Triton when it is installed and the codec runs in bfloat16 on CUDA.
    """
    from vllm_omni.model_executor.models.haru import codec_triton

    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"unknown codec backend {backend!r}")
    usable = codec_triton.is_available() and codec.device.type == "cuda" and codec.dtype == torch.bfloat16
    if backend == "triton" and not usable:
        raise RuntimeError("the Triton codec backend needs triton and a bfloat16 codec on CUDA")
    if backend != "torch" and usable:
        return codec_triton.TritonCodecStream(codec, batch)
    return CodecStream(codec, batch)
