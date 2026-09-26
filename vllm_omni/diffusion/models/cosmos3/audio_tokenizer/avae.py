# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusers-format AVAE audio tokenizer used by Cosmos3 sound generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import weight_norm
from vllm.logger import init_logger

from vllm_omni.diffusion.models.progress_bar import _is_rank_zero

logger = init_logger(__name__)


def _default_avae_config(
    *,
    sample_rate: int,
    audio_channels: int,
    io_channels: int,
    hop_size: int,
) -> dict[str, Any]:
    return {
        "sampling_rate": sample_rate,
        "hop_size": hop_size,
        "dec_dim": 320,
        "dec_c_mults": [1, 2, 4, 8, 16],
        "dec_strides": [2, 4, 5, 6, 8],
        "dec_out_channels": audio_channels,
        "vocoder_input_dim": io_channels,
        "normalization_type": "none",
        "normalize_latents": False,
        "tanh_input_scale": 1.5,
        "tanh_output_scale": 3.5,
        "tanh_clamp": 0.995,
    }


def _config_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def _load_config(
    config_path: str | Path | None,
    *,
    sample_rate: int,
    audio_channels: int,
    io_channels: int,
    hop_size: int,
) -> dict[str, Any]:
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise TypeError(f"Cosmos3 AVAE config must be a JSON object, got {type(config)!r}.")
        return config
    return _default_avae_config(
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        io_channels=io_channels,
        hop_size=hop_size,
    )


def _load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading AVAE .safetensors checkpoints requires safetensors.") from exc
        checkpoint = load_file(str(path), device=str(map_location))
    else:
        checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"AVAE checkpoint must be a flat state dict, got {type(checkpoint)!r}.")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError("AVAE checkpoint must be a flat tensor state dict.")
    return checkpoint


def _validate_diffusers_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    if not state_dict:
        raise RuntimeError("AVAE checkpoint is empty.")

    if not any(key.startswith("decoder.") for key in state_dict):
        raise RuntimeError("Cosmos3 AVAE checkpoint must contain diffusers-format decoder.* keys.")


class Snake1d(nn.Module):
    """One-dimensional Snake activation matching diffusers' Oobleck layout."""

    def __init__(self, hidden_dim: int, logscale: bool = True) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        alpha = torch.exp(self.alpha) if self.logscale else self.alpha
        beta = torch.exp(self.beta) if self.logscale else self.beta
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class OobleckResidualUnit(nn.Module):
    """Residual unit used by the diffusers Oobleck decoder."""

    def __init__(self, dimension: int = 16, dilation: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output_tensor = self.conv1(self.snake1(hidden_state))
        output_tensor = self.conv2(self.snake2(output_tensor))
        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + output_tensor


class OobleckDecoderBlock(nn.Module):
    """Decoder block used by the diffusers Oobleck decoder."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, output_padding: int = 0) -> None:
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        return self.res_unit3(hidden_state)


class OobleckDecoder(nn.Module):
    """Diffusers-compatible Oobleck decoder for Cosmos3 AVAE latents."""

    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list[int],
        channel_multiples: list[int],
    ) -> None:
        super().__init__()
        strides = upsampling_ratios
        channel_multiples = [1] + channel_multiples

        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))

        block = []
        for stride_index, stride in enumerate(strides):
            block.append(
                OobleckDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            )
        self.block = nn.ModuleList(block)
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


class Cosmos3AVAEAudioTokenizer(nn.Module):
    """Decoder-only AVAE tokenizer for Cosmos3 audio latents."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        sample_rate: int = 48000,
        audio_channels: int = 2,
        io_channels: int = 64,
        hop_size: int = 1920,
        normalize_latents: bool = False,
        normalization_type: str = "none",
        tanh_input_scale: float = 1.5,
        tanh_output_scale: float = 3.5,
        tanh_clamp: float = 0.995,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = torch.device(device)

        config = _load_config(
            config_path,
            sample_rate=sample_rate,
            audio_channels=audio_channels,
            io_channels=io_channels,
            hop_size=hop_size,
        )
        self.sample_rate = int(_config_get(config, "sampling_rate", "sample_rate", default=sample_rate))
        self.audio_channels = int(
            _config_get(
                config,
                "dec_out_channels",
                "audio_channels",
                default=2 if bool(config.get("stereo", audio_channels == 2)) else 1,
            )
        )
        self.latent_ch = int(_config_get(config, "vocoder_input_dim", "io_channels", "latent_ch", default=io_channels))
        dec_strides = [int(stride) for stride in _config_get(config, "dec_strides", default=[2, 4, 5, 6, 8])]
        self.hop_size = int(
            _config_get(config, "hop_size", default=math.prod(dec_strides) if dec_strides else hop_size)
        )
        dec_stride_product = math.prod(dec_strides)
        if dec_stride_product != self.hop_size:
            raise ValueError(
                "Cosmos3 AVAE config dec_strides product must equal hop_size "
                f"for correct latent/audio duration math: product={dec_stride_product}, hop_size={self.hop_size}."
            )

        normalization_type = str(_config_get(config, "normalization_type", default=normalization_type))
        normalize_latents = bool(_config_get(config, "normalize_latents", default=normalize_latents))
        if normalization_type == "none" and normalize_latents:
            normalization_type = "tanh"
        self.normalization_type = normalization_type
        self.tanh_input_scale = float(_config_get(config, "tanh_input_scale", default=tanh_input_scale))
        self.tanh_output_scale = float(_config_get(config, "tanh_output_scale", default=tanh_output_scale))
        self.tanh_clamp = float(_config_get(config, "tanh_clamp", default=tanh_clamp))

        self.decoder = OobleckDecoder(
            channels=int(_config_get(config, "dec_dim", default=320)),
            input_channels=self.latent_ch,
            audio_channels=self.audio_channels,
            upsampling_ratios=list(reversed(dec_strides)),
            channel_multiples=list(_config_get(config, "dec_c_mults", default=[1, 2, 4, 8, 16])),
        )
        state_dict = _load_checkpoint(checkpoint_path, self.device)
        _validate_diffusers_state_dict(state_dict)

        # The checkpoint also contains encoder weights, which we do not support here, hence strict=False
        self.load_state_dict(state_dict, strict=False)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.to(device=self.device, dtype=self.dtype)
        if _is_rank_zero():
            logger.info("Loaded diffusers-format Cosmos3 AVAE checkpoint from %s", checkpoint_path)

    @property
    def temporal_compression_factor(self) -> int:
        return self.hop_size

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(num_audio_samples) // self.temporal_compression_factor

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(num_latent_samples) * self.temporal_compression_factor

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.normalization_type == "tanh":
            in_dtype = latent.dtype
            latent = torch.clamp(
                latent.float() / self.tanh_output_scale,
                -self.tanh_clamp,
                self.tanh_clamp,
            )
            return (torch.atanh(latent) * self.tanh_input_scale).to(in_dtype)
        if self.normalization_type != "none":
            raise ValueError(f"Unsupported AVAE normalization_type={self.normalization_type!r}.")
        return latent

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, force_pad: bool = False) -> torch.Tensor:
        del audio, force_pad
        raise NotImplementedError("Cosmos3AVAEAudioTokenizer is decoder-only for diffusers-format sound_tokenizer/.")

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        in_dtype = latent.dtype
        squeeze = latent.ndim == 2
        if squeeze:
            latent = latent.unsqueeze(0)
        z = self._denormalize_latent(latent.to(self.device)).to(self.dtype)
        audio = self.decoder(z).clamp(-1.0, 1.0).to(in_dtype)
        return audio.squeeze(0) if squeeze else audio
