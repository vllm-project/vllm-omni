# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native packing helpers for the MAGI-2 Preview transformer.

The released model consumes one varlen sequence per sample.  Tokens are laid
out in VIDEO -> AUDIO -> TEXT order, followed by zero or more reference-image
special-token/image-token pairs.  This module keeps that layout and its 9-D
coordinate metadata independent of any particular distributed topology.

The packing math is adapted from SandAI's Apache-2.0 MAGI-2 Preview inference
implementation.  It intentionally has no dependency on that implementation at
runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from typing import Literal

import torch
from einops import rearrange
from torch.nn import functional as F

from .attention import VarlenHandler


class Modality(IntEnum):
    """Checkpoint modality IDs used by the pre/post adapters."""

    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    # TIME is remapped to TEXT by the transformer after selecting time tokens.
    TIME = 3


@dataclass(frozen=True)
class Magi2PreviewDataProxyConfig:
    """Packing configuration for the released preview checkpoint."""

    t_patch_size: int = 1
    patch_size: int = 1
    spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
    add_time_token: bool = False
    time_channel_dim: int = 64
    time_aligned_rope: bool = False
    audio_latent_fps: float = 25.0
    time_pos_fps: float = 3.125
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0

    def __post_init__(self) -> None:
        if self.t_patch_size <= 0 or self.patch_size <= 0:
            raise ValueError("MAGI-2 patch sizes must be positive")
        if self.time_channel_dim < 0:
            raise ValueError("time_channel_dim must be non-negative")


@dataclass
class ModelInput:
    """Unpacked tensors for one native preview-transformer invocation."""

    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | Sequence[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | Sequence[int]
    t: torch.Tensor
    ref_audio_feat: torch.Tensor | None = None
    ref_audio_feat_len: torch.Tensor | Sequence[int] | None = None
    ref_video_feat: torch.Tensor | None = None
    ref_video_feat_len: torch.Tensor | Sequence[int] | None = None
    per_token_video_t: torch.Tensor | None = None
    per_token_audio_t: torch.Tensor | None = None
    # Images are [B, M, C, 1, H, W], lengths are [B, M, 2] holding
    # the latent patch-grid (H, W), and special tokens are [B, M, D].
    ref_image_feat: torch.Tensor | None = None
    ref_image_feat_len: torch.Tensor | None = None
    ref_image_special_token_embedding: torch.Tensor | None = None


def _to_int(value: int | torch.Tensor) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


def _length_at(value: torch.Tensor | Sequence[int], index: int) -> int:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        if index:
            raise IndexError("a scalar length only describes one sample")
        return _to_int(value)
    return _to_int(value[index])


def _pad_cat(tensors: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not tensors:
        return torch.empty(0, 0, device=device, dtype=dtype)
    max_channel = max(tensor.shape[-1] for tensor in tensors)
    return torch.cat(
        [F.pad(tensor.to(device=device, dtype=dtype), (0, max_channel - tensor.shape[-1])) for tensor in tensors],
        dim=0,
    )


def _segment_paint(
    values: list[int],
    seqlens: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    output_size: int,
) -> torch.Tensor:
    mapping = torch.empty(output_size, dtype=dtype, device=device)
    offset = 0
    for value, seqlen in zip(values, seqlens, strict=True):
        mapping[offset : offset + seqlen] = int(value)
        offset += seqlen
    if offset != output_size:
        raise ValueError(f"segment lengths total {offset}, expected {output_size}")
    return mapping


def _seqlens_to_cu_seqlens(seqlens: list[int], device: torch.device | None = None) -> torch.Tensor:
    lengths = torch.tensor(seqlens, dtype=torch.int32, device=device)
    return F.pad(torch.cumsum(lengths, dim=0), (1, 0))


def _len_to_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().to(torch.long).reshape(-1).tolist()]


def get_coords(
    shape: tuple[int, int, int],
    ref_feat_shape: tuple[int, int, int],
    offset_thw: tuple[int, int, int] = (0, 0, 0),
    *,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    time_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build MAGI-2's ``(t,h,w,T,H,W,ref_T,ref_H,ref_W)`` rows."""

    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape
    offset_t, offset_h, offset_w = offset_thw

    if time_positions is None:
        time_range = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    else:
        if time_positions.numel() != ori_t:
            raise ValueError(f"received {time_positions.numel()} time positions for {ori_t} time steps")
        time_range = time_positions.to(device=device, dtype=dtype) + offset_t
    height_range = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_range = torch.arange(ori_w, device=device, dtype=dtype) + offset_w
    time_grid, height_grid, width_grid = torch.meshgrid(
        time_range,
        height_range,
        width_range,
        indexing="ij",
    )
    coords = torch.stack((time_grid, height_grid, width_grid), dim=-1).reshape(-1, 3)
    metadata = torch.tensor(
        (ori_t, ori_h, ori_w, ref_t, ref_h, ref_w),
        device=device,
        dtype=dtype,
    )
    return torch.cat((coords, metadata.expand(coords.shape[0], -1)), dim=-1)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """The preview checkpoint's FP32 diffusion-time channel embedding."""

    position = position.to(torch.float32) * 1000.0
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=position.device) / half
    )
    arguments = position[:, None].float() * frequencies[None]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def _patchify_3d(x: torch.Tensor, t_patch_size: int, patch_size: int) -> torch.Tensor:
    """Non-overlapping 3-D unfold with the released channel-major order."""

    kernel = (t_patch_size, patch_size, patch_size)
    if x.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
    if not all(size >= kernel_size for size, kernel_size in zip(x.shape[2:], kernel, strict=True)):
        return torch.empty(
            x.shape[0],
            0,
            x.shape[1] * math.prod(kernel),
            device=x.device,
            dtype=x.dtype,
        )

    patches = x.unfold(2, t_patch_size, t_patch_size)
    patches = patches.unfold(3, patch_size, patch_size)
    patches = patches.unfold(4, patch_size, patch_size)
    # [B,C,T,H,W,pT,pH,pW] -> [B,T*H*W,C*pT*pH*pW]
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    return patches.reshape(x.shape[0], -1, x.shape[1] * math.prod(kernel))


@dataclass
class SingleData:
    """Packed metadata for one sample."""

    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    diffusion_t: torch.Tensor | None = None
    per_token_video_t: torch.Tensor | None = None
    per_token_audio_t: torch.Tensor | None = None
    time_channel_dim: int = 0
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0
    time_pos_fps: float = 3.125
    ref_image_feats: list[torch.Tensor] | None = None
    ref_image_feat_lens: list[list[int]] | None = None
    ref_image_special_tokens: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        self.video_token_num = self.video_x_t.shape[0]
        self.origin_audio_feat_len = self.audio_x_t.shape[0]
        self.audio_x_t = self.audio_x_t[: self.audio_feat_len]
        self.txt_feat = self.txt_feat[: self.txt_feat_len]
        if self.per_token_audio_t is not None:
            self.per_token_audio_t = self.per_token_audio_t[: self.audio_feat_len]

        self.ref_image_feats = self.ref_image_feats or []
        self.ref_image_feat_lens = self.ref_image_feat_lens or []
        self.ref_image_special_tokens = self.ref_image_special_tokens or []
        if not (len(self.ref_image_feats) == len(self.ref_image_feat_lens) == len(self.ref_image_special_tokens)):
            raise ValueError("reference-image features, lengths, and special tokens must have equal counts")
        self.ref_image_token_nums = [math.prod(feat_len) for feat_len in self.ref_image_feat_lens]
        self.ref_image_feats = [
            feature[:token_num]
            for feature, token_num in zip(self.ref_image_feats, self.ref_image_token_nums, strict=True)
        ]
        self.num_ref_images = len(self.ref_image_feats)
        self.total_ref_image_feat_len = sum(self.ref_image_token_nums)
        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.video_x_t.device

    @property
    def default_dtype(self) -> torch.dtype:
        return self.video_x_t.dtype

    @property
    def add_time_token(self) -> bool:
        return self.diffusion_t is not None

    @property
    def total_token_num(self) -> int:
        total = self.video_token_num + self.audio_feat_len + self.txt_feat_len
        total += self.total_ref_image_feat_len + self.num_ref_images
        return total + int(self.add_time_token)

    @property
    def feat_to_cat(self) -> list[torch.Tensor]:
        tensors = [self.video_x_t, self.audio_x_t, self.txt_feat]
        for index in range(self.num_ref_images):
            tensors.append(self.ref_image_special_tokens[index])
            tensors.append(self.ref_image_feats[index])
        if self.add_time_token:
            tensors.append(self.diffusion_t.to(device=self.device, dtype=self.default_dtype).reshape(1, 1))
        return tensors

    @property
    def token_sequence(self) -> torch.Tensor:
        return _pad_cat(self.feat_to_cat, self.device, self.default_dtype)

    @property
    def modality_map_seqlens(self) -> tuple[list[int], list[int]]:
        seqlens = [self.video_token_num, self.audio_feat_len, self.txt_feat_len]
        modalities = [Modality.VIDEO, Modality.AUDIO, Modality.TEXT]
        for index in range(self.num_ref_images):
            seqlens.extend((1, self.ref_image_token_nums[index]))
            modalities.extend((Modality.TEXT, Modality.VIDEO))
        if self.add_time_token:
            seqlens.append(1)
            modalities.append(Modality.TIME)
        return seqlens, modalities

    @property
    def modality_mapping(self) -> torch.Tensor:
        seqlens, modalities = self.modality_map_seqlens
        return _segment_paint(
            modalities,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    def _default_coords(
        self,
        shape: tuple[int, int, int],
        ref_feat_shape: tuple[int, int, int],
        offset_thw: tuple[int, int, int] = (0, 0, 0),
        time_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return get_coords(
            shape,
            ref_feat_shape,
            offset_thw,
            device=self.device,
            dtype=self.default_dtype,
            time_positions=time_positions,
        )

    @property
    def coords_to_cat(self) -> list[torch.Tensor]:
        t_steps = self.t // self.t_patch_size
        h_steps = self.h // self.patch_size
        w_steps = self.w // self.patch_size
        if self.spatial_rope_interpolation == "inter":
            video_h_ref, video_w_ref = 32, 32
        else:
            video_h_ref, video_w_ref = h_steps, w_steps

        video_coords = self._default_coords(
            (t_steps, h_steps, w_steps),
            (t_steps, video_h_ref, video_w_ref),
        )
        magic_audio_ref_t = (self.audio_feat_len - 1) // 8 + 1
        audio_coords = self._default_coords(
            (self.audio_feat_len, 1, 1),
            (magic_audio_ref_t // self.t_patch_size, 1, 1),
        )
        coords = [
            video_coords,
            audio_coords,
            self._default_coords(
                (self.txt_feat_len, 1, 1),
                (1, 1, 1),
                offset_thw=(-self.txt_feat_len, 0, 0),
            ),
        ]

        for index in range(self.num_ref_images):
            token_len = self.ref_image_token_nums[index]
            if len(self.ref_image_feat_lens[index]) >= 2:
                image_h, image_w = self.ref_image_feat_lens[index][:2]
            else:
                image_h = image_w = math.ceil(math.sqrt(token_len))
            # Keep image context after the generated clip with a one-step gap.
            time_offset = t_steps + 2 + index
            coords.append(
                torch.tensor(
                    [[time_offset, -1, -1, 1, image_h, image_w, 1, image_h, image_w]],
                    device=self.device,
                    dtype=self.default_dtype,
                )
            )
            coords.append(
                self._default_coords(
                    (1, image_h, image_w),
                    (1, image_h, image_w),
                    offset_thw=(time_offset, 0, 0),
                )[:token_len]
            )

        if self.add_time_token:
            coords.append(self._default_coords((1, 1, 1), (1, 1, 1))[:1])
        return coords

    @property
    def coords_mapping(self) -> torch.Tensor:
        return torch.cat(self.coords_to_cat, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        if self.time_channel_dim == 0:
            return torch.empty(self.total_token_num, 0, device=self.device)
        if self.per_token_video_t is None or self.per_token_audio_t is None:
            raise ValueError("per-token video and audio timesteps are required when time_channel_dim is non-zero")
        parts = [
            self.per_token_video_t.squeeze(-1),
            self.per_token_audio_t.squeeze(-1),
            torch.zeros(self.txt_feat_len, device=self.device),
        ]
        for index in range(self.num_ref_images):
            parts.append(torch.zeros(1, device=self.device))
            parts.append(torch.zeros(self.ref_image_token_nums[index], device=self.device))
        if self.add_time_token:
            parts.append(self.diffusion_t.reshape(1).to(self.device))
        raw_t = torch.cat(parts, dim=0)
        if self.time_channel_dim == 1:
            return raw_t.unsqueeze(-1)
        return sinusoidal_embedding_1d(self.time_channel_dim, raw_t)


@dataclass
class SimplePackedData:
    """A concatenation of independently addressable packed samples."""

    items: list[SingleData]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("SimplePackedData must contain at least one item")
        self._total_token_num_list = [item.total_token_num for item in self.items]
        self._total_token_num_sum = sum(self._total_token_num_list)
        self._total_token_num_max = max(self._total_token_num_list)
        self._total_token_cu_seqlens = _seqlens_to_cu_seqlens(self._total_token_num_list)

    @property
    def device(self) -> torch.device:
        return self.items[0].device

    @property
    def default_dtype(self) -> torch.dtype:
        return self.items[0].default_dtype

    @property
    def token_sequence(self) -> torch.Tensor:
        tensors = list(chain.from_iterable(item.feat_to_cat for item in self.items))
        return _pad_cat(tensors, self.device, self.default_dtype)

    @property
    def modality_mapping(self) -> torch.Tensor:
        seqlens: list[int] = []
        modalities: list[int] = []
        for item in self.items:
            item_seqlens, item_modalities = item.modality_map_seqlens
            seqlens.extend(item_seqlens)
            modalities.extend(item_modalities)
        return _segment_paint(
            modalities,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    @property
    def coords_mapping(self) -> torch.Tensor:
        coords = list(chain.from_iterable(item.coords_to_cat for item in self.items))
        return torch.cat(coords, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        return torch.cat([item.time_token_sequence for item in self.items], dim=0)

    @property
    def total_token_num(self) -> int:
        return self._total_token_num_sum

    @property
    def cu_seqlen(self) -> torch.Tensor:
        return self._total_token_cu_seqlens.clone()

    @property
    def max_seqlen(self) -> int:
        return self._total_token_num_max

    def __getitem__(self, index: int) -> SingleData:
        return self.items[index]

    def depack_token_sequence(self, token_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if token_sequence.shape[0] != self.total_token_num:
            raise ValueError(f"model returned {token_sequence.shape[0]} tokens, expected {self.total_token_num}")
        videos: list[torch.Tensor] = []
        audios: list[torch.Tensor] = []
        token_slices = torch.split(token_sequence, self._total_token_num_list, dim=0)
        for item, token_slice in zip(self.items, token_slices, strict=True):
            video_flat = token_slice[: item.video_token_num, : item.video_channel]
            output_channels = item.video_channel // (item.t_patch_size * item.patch_size * item.patch_size)
            video = rearrange(
                video_flat,
                "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
                T=item.t // item.t_patch_size,
                H=item.h // item.patch_size,
                W=item.w // item.patch_size,
                pT=item.t_patch_size,
                pH=item.patch_size,
                pW=item.patch_size,
                C=output_channels,
            ).contiguous()
            audio = torch.zeros(
                item.origin_audio_feat_len,
                item.audio_channel,
                device=token_sequence.device,
                dtype=token_sequence.dtype,
            )
            audio[: item.audio_feat_len] = token_slice[
                item.video_token_num : item.video_token_num + item.audio_feat_len,
                : item.audio_channel,
            ]
            videos.append(video)
            audios.append(audio)
        return torch.stack(videos, dim=0), torch.stack(audios, dim=0)


@dataclass(frozen=True)
class PackedModelInput:
    """Request-owned packed transformer arguments and output layout."""

    token_sequence: torch.Tensor
    coords_mapping: torch.Tensor
    modality_mapping: torch.Tensor
    varlen_handler: VarlenHandler
    time_token_sequence: torch.Tensor
    output_layout: SimplePackedData

    @property
    def model_args(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, VarlenHandler, torch.Tensor]:
        return (
            self.token_sequence,
            self.coords_mapping,
            self.modality_mapping,
            self.varlen_handler,
            self.time_token_sequence,
        )


class Magi2DataProxy:
    """Convert dense modality tensors to/from the transformer's packed ABI.

    The proxy is request-stateless: all packing state needed by
    ``process_output`` is returned as ``PackedModelInput.output_layout`` rather
    than stored on this shared instance, so concurrent requests cannot
    overwrite one another.
    """

    def __init__(self, config: Magi2PreviewDataProxyConfig | None = None) -> None:
        self.config = config or Magi2PreviewDataProxyConfig()
        self.patch_size = self.config.patch_size
        self.t_patch_size = self.config.t_patch_size

    def img2tokens(self, x_t: torch.Tensor) -> torch.Tensor:
        return _patchify_3d(x_t, self.t_patch_size, self.patch_size)

    @staticmethod
    def _validate_batch(data: ModelInput) -> None:
        if data.x_t.ndim != 5:
            raise ValueError(f"x_t must be [B,C,T,H,W], got {tuple(data.x_t.shape)}")
        if data.audio_x_t.ndim != 3 or data.txt_feat.ndim != 3:
            raise ValueError("audio_x_t and txt_feat must be [B,L,C]")
        batch_size = data.x_t.shape[0]
        if data.audio_x_t.shape[0] != batch_size or data.txt_feat.shape[0] != batch_size:
            raise ValueError("video, audio, and text batch sizes must match")

        ref_values = (
            data.ref_image_feat,
            data.ref_image_feat_len,
            data.ref_image_special_token_embedding,
        )
        if any(value is not None for value in ref_values) and not all(value is not None for value in ref_values):
            raise ValueError("reference-image features, lengths, and special tokens must be provided together")
        if data.ref_image_feat is not None:
            if data.ref_image_feat.ndim != 6:
                raise ValueError("ref_image_feat must be [B,M,C,T,H,W]")
            if data.ref_image_feat_len.ndim != 3 or data.ref_image_feat_len.shape[-1] < 2:
                raise ValueError("ref_image_feat_len must be [B,M,2]")
            if data.ref_image_special_token_embedding.ndim != 3:
                raise ValueError("ref_image_special_token_embedding must be [B,M,D]")
            expected = data.ref_image_feat.shape[:2]
            if data.ref_image_feat_len.shape[:2] != expected:
                raise ValueError("reference-image feature and length batch/image counts differ")
            if data.ref_image_special_token_embedding.shape[:2] != expected:
                raise ValueError("reference-image feature and special-token batch/image counts differ")

    def process_input(
        self,
        data: ModelInput,
    ) -> PackedModelInput:
        self._validate_batch(data)
        batch_size, _, video_t, video_h, video_w = data.x_t.shape
        video_tokens = self.img2tokens(data.x_t)
        audio_tokens = data.audio_x_t.contiguous()
        text_tokens = data.txt_feat.contiguous()

        per_token_video_tokens = None
        per_token_audio = None
        if self.config.time_channel_dim > 0 and data.per_token_video_t is not None:
            per_token_video_tokens = self.img2tokens(data.per_token_video_t)[:, :, :1]
            per_token_audio = data.per_token_audio_t

        has_ref_image = data.ref_image_feat is not None
        num_ref_images = data.ref_image_feat.shape[1] if has_ref_image else 0
        ref_device, ref_dtype = text_tokens.device, text_tokens.dtype

        items: list[SingleData] = []
        for batch_index in range(batch_size):
            ref_image_feats: list[torch.Tensor] = []
            ref_image_feat_lens: list[list[int]] = []
            ref_image_special_tokens: list[torch.Tensor] = []
            for image_index in range(num_ref_images):
                feature = self.img2tokens(data.ref_image_feat[batch_index, image_index].unsqueeze(0)).squeeze(0)
                ref_image_feats.append(feature)
                ref_image_feat_lens.append(_len_to_list(data.ref_image_feat_len[batch_index, image_index]))
                ref_image_special_tokens.append(
                    data.ref_image_special_token_embedding[batch_index, image_index]
                    .to(device=ref_device, dtype=ref_dtype)
                    .unsqueeze(0)
                )

            if isinstance(data.t, torch.Tensor) and data.t.ndim == 0:
                diffusion_t = data.t
            else:
                diffusion_t = data.t[batch_index]
            items.append(
                SingleData(
                    video_x_t=video_tokens[batch_index],
                    audio_x_t=audio_tokens[batch_index],
                    audio_feat_len=_length_at(data.audio_feat_len, batch_index),
                    txt_feat=text_tokens[batch_index],
                    txt_feat_len=_length_at(data.txt_feat_len, batch_index),
                    t=video_t,
                    h=video_h,
                    w=video_w,
                    patch_size=self.patch_size,
                    t_patch_size=self.t_patch_size,
                    spatial_rope_interpolation=self.config.spatial_rope_interpolation,
                    diffusion_t=diffusion_t if self.config.add_time_token else None,
                    per_token_video_t=(
                        per_token_video_tokens[batch_index] if per_token_video_tokens is not None else None
                    ),
                    per_token_audio_t=(per_token_audio[batch_index] if per_token_audio is not None else None),
                    time_channel_dim=self.config.time_channel_dim,
                    time_pos_fps=self.config.time_pos_fps,
                    vae_first_latent_is_image=self.config.vae_first_latent_is_image,
                    video_fps=self.config.video_fps,
                    ref_image_feats=ref_image_feats,
                    ref_image_feat_lens=ref_image_feat_lens,
                    ref_image_special_tokens=ref_image_special_tokens,
                )
            )

        packed = SimplePackedData(items)
        cu_seqlens = packed.cu_seqlen.to(device=data.x_t.device, dtype=torch.int32)
        varlen_handler = VarlenHandler(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=packed.max_seqlen,
            max_seqlen_k=packed.max_seqlen,
        )
        return PackedModelInput(
            token_sequence=packed.token_sequence,
            coords_mapping=packed.coords_mapping,
            modality_mapping=packed.modality_mapping,
            varlen_handler=varlen_handler,
            time_token_sequence=packed.time_token_sequence,
            output_layout=packed,
        )

    @staticmethod
    def process_output(
        x: torch.Tensor,
        output_layout: SimplePackedData,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return output_layout.depack_token_sequence(x)


__all__ = [
    "Magi2DataProxy",
    "Magi2PreviewDataProxyConfig",
    "Modality",
    "ModelInput",
    "PackedModelInput",
    "SimplePackedData",
    "SingleData",
    "VarlenHandler",
    "get_coords",
    "sinusoidal_embedding_1d",
]
