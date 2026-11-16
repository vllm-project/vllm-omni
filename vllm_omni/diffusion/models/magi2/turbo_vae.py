# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native decoder for the TurboVAED checkpoint shipped with MAGI-2.

Adapted from SandAI's MAGI-2 Preview inference implementation. Training-only
modules and MagiCompiler integration are intentionally omitted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

_WAN22_LATENT_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
_WAN22_LATENT_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


def turbo_unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.ndim != 5:
        raise ValueError(f"Expected a 5-D TurboVAED output, got {tuple(x.shape)}")
    batch, patched_channels, frames, height, width = x.shape
    patch_area = patch_size * patch_size
    if patched_channels % patch_area:
        raise ValueError("TurboVAED output channels must be divisible by patch_size squared")
    channels = patched_channels // patch_area
    x = x.view(batch, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(batch, channels, frames, height * patch_size, width * patch_size)


class _RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.float().pow(2).mean(1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(dtype)


class _CausalConv3d(nn.Module):
    """Checkpoint-compatible non-causal 3-D convolution with edge time padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        *,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if is_causal:
            raise ValueError("The released MAGI-2 TurboVAED checkpoint is non-causal")
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            padding=(0, self.kernel_size[1] // 2, self.kernel_size[2] // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_pad = (self.kernel_size[0] - 1) // 2
        if temporal_pad:
            x = torch.cat(
                [
                    x[:, :, :1].repeat(1, 1, temporal_pad, 1, 1),
                    x,
                    x[:, :, -1:].repeat(1, 1, temporal_pad, 1, 1),
                ],
                dim=2,
            )
        return self.conv(x)


class _ResNetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        is_upsampler_modified: bool = False,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.nonlinearity = nn.SiLU()
        self.replace_nonlinearity = nn.ReLU()
        self.is_upsampler_modified = is_upsampler_modified
        self.norm1 = _RMSNorm()
        self.conv1 = _CausalConv3d(in_channels, out_channels, is_causal=is_causal)
        self.norm2 = _RMSNorm()
        self.conv2 = _CausalConv3d(out_channels, out_channels, is_causal=is_causal)
        self.norm3 = _RMSNorm(eps=1e-6) if in_channels != out_channels else None
        self.conv_shortcut = (
            _CausalConv3d(in_channels, out_channels, kernel_size=1, is_causal=is_causal)
            if in_channels != out_channels
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inputs)
        x = self.replace_nonlinearity(x) if self.is_upsampler_modified else self.nonlinearity(x)
        x = self.conv1(x)
        x = self.conv2(self.nonlinearity(self.norm2(x)))
        residual = self.norm3(inputs) if self.norm3 is not None else inputs
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return x + residual


class _WanUpsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class _WanResample(nn.Module):
    def __init__(self, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                _WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
        else:
            self.resample = nn.Identity()
        if mode == "upsample3d":
            self.time_conv = _CausalConv3d(channels, channels * 2, (3, 1, 1))

    def forward(self, x: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
        batch, channels, frames, _height, _width = x.shape
        if self.mode == "upsample3d":
            x = self.time_conv(x)
            x = rearrange(x, "b (split c) t h w -> b c (t split) h w", split=2)
            if x.shape[1:3] != (channels, frames * 2):
                raise RuntimeError("Unexpected temporal upsample shape")
            if is_first_chunk:
                x = x[:, :, 1:]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=batch)


class _MidBlock3d(nn.Module):
    def __init__(self, channels: int, num_layers: int, *, is_causal: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([_ResNetBlock3d(channels, is_causal=is_causal) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return x


class _UpBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        *,
        spatio_temporal_scale: bool,
        spatio_only: bool,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.conv_in = (
            _ResNetBlock3d(in_channels, out_channels, is_causal=is_causal) if in_channels != out_channels else None
        )
        self.upsamplers = (
            nn.ModuleList([_WanResample(out_channels, "upsample2d" if spatio_only else "upsample3d")])
            if spatio_temporal_scale
            else None
        )
        self.resnets = nn.ModuleList(
            [
                _ResNetBlock3d(
                    out_channels,
                    is_upsampler_modified=spatio_temporal_scale,
                    is_causal=is_causal,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x, is_first_chunk=is_first_chunk)
        for resnet in self.resnets:
            x = resnet(x)
        return x


class _TurboDecoder3d(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: tuple[int, ...],
        spatio_temporal_scaling: tuple[bool, ...],
        spatio_only: tuple[bool, ...],
        patch_size: int,
        decoder_causal: bool,
        use_unpatchify: bool,
    ) -> None:
        super().__init__()
        if patch_size != 2 or not use_unpatchify:
            raise ValueError("The released MAGI-2 TurboVAED requires patch_size=2 and use_unpatchify=true")
        self.patch_size = patch_size
        blocks = tuple(reversed(block_out_channels))
        scaling = tuple(reversed(spatio_temporal_scaling))
        layers = tuple(reversed(layers_per_block))
        spatial_only = tuple(reversed(spatio_only))
        current = blocks[0]

        self.conv_in = _CausalConv3d(latent_channels, current, is_causal=decoder_causal)
        self.mid_block = _MidBlock3d(current, layers[0], is_causal=decoder_causal)
        self.up_blocks = nn.ModuleList()
        for index, output in enumerate(blocks):
            self.up_blocks.append(
                _UpBlock3d(
                    current,
                    output,
                    layers[index + 1],
                    spatio_temporal_scale=scaling[index],
                    spatio_only=spatial_only[index],
                    is_causal=decoder_causal,
                )
            )
            current = output
        self.conv_act = nn.SiLU()
        self.conv_out = _CausalConv3d(
            current,
            out_channels * patch_size * patch_size,
            is_causal=decoder_causal,
        )

    def forward(self, x: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x, is_first_chunk=is_first_chunk)
        dtype = x.dtype
        x = (x * torch.rsqrt(x.float().pow(2).mean(1, keepdim=True) + 1e-8)).to(dtype)
        return turbo_unpatchify(self.conv_out(self.conv_act(x)), self.patch_size)


def _load_turbo_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"TurboVAED config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    unsupported = {
        "decoder_causal": config.get("decoder_causal", False),
        "decoder_is_dw_conv": any(config.get("decoder_is_dw_conv", [])),
        "use_unpatchify": not config.get("use_unpatchify", False),
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError(f"Unsupported TurboVAED configuration: {', '.join(enabled)}")
    return config


def extract_turbo_decoder_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Select decoder EMA tensors and strip the training module prefix."""

    state: Any = checkpoint
    if "ema_state_dict" in state:
        state = state["ema_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("TurboVAED checkpoint does not contain a state dictionary")
    result = {}
    for key, value in state.items():
        target = key[7:] if key.startswith("module.") else key
        if target.startswith("decoder."):
            result[target] = value
    if not result:
        raise ValueError("TurboVAED checkpoint contains no decoder tensors")
    return result


class Magi2TurboVAEDecoder(nn.Module, DistributedVaeMixin):
    """Load and run MAGI-2's default distilled video decoder."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        config = _load_turbo_config(Path(config_path))
        self.z_dim = int(config["latent_channels"])
        self.first_chunk_size = int(config["first_chunk_size"])
        self.step_size = int(config["step_size"])
        self.temporal_compression_ratio = int(config["temporal_compression_ratio"])
        self.spatial_compression_ratio = int(config["spatial_compression_ratio"])
        self.decoder = _TurboDecoder3d(
            latent_channels=self.z_dim,
            out_channels=int(config["out_channels"]),
            block_out_channels=tuple(config["decoder_block_out_channels"]),
            layers_per_block=tuple(config["decoder_layers_per_block"]),
            spatio_temporal_scaling=tuple(config["decoder_spatio_temporal_scaling"]),
            spatio_only=tuple(config["decoder_spatio_only"]),
            patch_size=int(config["patch_size"]),
            decoder_causal=bool(config["decoder_causal"]),
            use_unpatchify=bool(config["use_unpatchify"]),
        ).to(device=device, dtype=dtype)
        self.register_buffer(
            "latent_mean",
            torch.tensor(_WAN22_LATENT_MEAN, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            torch.tensor(_WAN22_LATENT_STD, dtype=torch.float32, device=device),
            persistent=False,
        )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"TurboVAED checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = extract_turbo_decoder_state_dict(checkpoint)
        self.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)
        self.use_tiling = False
        self.use_slicing = False
        if dist.is_available() and dist.is_initialized():
            self.init_distributed()

    def set_parallel_size(self, parallel_size: int, mode: str = "tile") -> None:
        if mode != "tile":
            raise ValueError(f"MAGI-2 TurboVAE supports temporal tile parallelism only, got {mode!r}")
        if not hasattr(self, "distributed_executor"):
            if parallel_size > 1:
                raise RuntimeError("TurboVAE patch parallelism requires initialized torch.distributed")
            return
        super().set_parallel_size(parallel_size, mode=mode)

    def is_distributed_enabled(self) -> bool:
        return hasattr(self, "distributed_executor") and super().is_distributed_enabled()

    def _prepare_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, int]:
        z = z.to(device=self.latent_std.device)
        dtype = z.dtype
        z = z * self.latent_std.view(1, self.z_dim, 1, 1, 1) + self.latent_mean.view(1, self.z_dim, 1, 1, 1)
        z = z.to(dtype)
        first = self.first_chunk_size
        step = self.step_size
        frames = z.shape[2]
        if frames < first:
            padding = first - frames
        elif (frames - first) % step:
            padding = step - (frames - first) % step
        else:
            padding = 0
        if padding:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, padding, 1, 1)], dim=2)
        return z, padding

    def _chunk_tasks(self, z: torch.Tensor, padding: int) -> tuple[list[TileTask], GridSpec]:
        first = self.first_chunk_size
        step = self.step_size
        overlap = self.temporal_compression_ratio
        frames = z.shape[2]
        descriptors: list[tuple[int, int, bool, int, int]] = []
        if frames == first:
            descriptors.append((0, frames, True, 0, 0))
        else:
            descriptors.append((0, first + 1, True, 0, overlap))
            for index in range(first, frames, step):
                last = index + step == frames
                descriptors.append(
                    (
                        index - 1,
                        index + step if last else index + step + 1,
                        False,
                        overlap,
                        0 if last else overlap,
                    )
                )

        tasks = [
            TileTask(
                tile_id=chunk_index,
                grid_coord=(chunk_index, int(is_first), crop_left, crop_right),
                tensor=z[:, :, left:right],
                workload=(right - left) * z.shape[3] * z.shape[4],
            )
            for chunk_index, (left, right, is_first, crop_left, crop_right) in enumerate(descriptors)
        ]
        return tasks, GridSpec(
            split_dims=(2,),
            grid_shape=(len(tasks),),
            tile_spec={"padding": padding},
            output_dtype=z.dtype,
        )

    def _decode_chunk(self, task: TileTask) -> torch.Tensor:
        _index, is_first, crop_left, crop_right = task.grid_coord
        assert isinstance(task.tensor, torch.Tensor)
        output = self.decoder(task.tensor, is_first_chunk=bool(is_first))
        right = output.shape[2] - crop_right if crop_right else output.shape[2]
        return output[:, :, crop_left:right]

    def _merge_chunks(
        self,
        chunks: dict[tuple[int, ...], torch.Tensor],
        grid_spec: GridSpec,
    ) -> torch.Tensor:
        output = torch.cat([tensor for _coord, tensor in sorted(chunks.items())], dim=2)
        padding = int(grid_spec.tile_spec["padding"])
        if padding:
            output = output[:, :, : -padding * self.temporal_compression_ratio]
        return output

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, *, output_offload: bool = False) -> torch.Tensor:
        z, padding = self._prepare_latent(z)
        tasks, grid_spec = self._chunk_tasks(z, padding)
        if self.is_distributed_enabled():
            output = self.distributed_executor.execute(
                z,
                DistributedOperator(
                    split=lambda _z: (tasks, grid_spec),
                    exec=self._decode_chunk,
                    merge=self._merge_chunks,
                ),
                broadcast_result=False,
            )
            return output.cpu() if output_offload and output.numel() else output

        chunks = []
        for task in tasks:
            tensor = self._decode_chunk(task)
            chunks.append((task.grid_coord, tensor.cpu() if output_offload else tensor))
        return self._merge_chunks(dict(chunks), grid_spec)

    forward = decode


__all__ = [
    "Magi2TurboVAEDecoder",
    "extract_turbo_decoder_state_dict",
    "turbo_unpatchify",
]
