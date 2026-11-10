# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2026 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import struct
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from diffusers.models.autoencoders.vae import DecoderOutput
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.activation import SiluAndMul
from vllm_omni.diffusion.models.hy_nomas.decoder_attention import packed_local_attention
from vllm_omni.diffusion.models.hy_nomas.decoder_kernels import apply_mrope, split_norm_rope

# Ported from jiaxiangc/diffusers at 12722ef11acf8e335f96cfd4faea17de78cd8909.
# Preserve checkpoint names, FP32 normalization, and absolute-frame decoder noise.


TUBELET = (4, 16, 16)
TUBE_DIM = TUBELET[0] * TUBELET[1] * TUBELET[2] * 3
LATENT_CHANNELS = 32
# A DCAE latent cell is (4, 32, 32) pixels: 1 tubelet in time, 2 x 2 in space.
LATENT_TOKEN_GRID = 2
_QK_CHUNK_INPUT_BYTES = 16 * 2**20


@dataclass(frozen=True)
class PixelUMMArchitecture:
    """Qwen3-8B (hidden 4096) with PixelUMM's SenseNova T/H/W MRoPE."""

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta_t: float = 1_000_000.0
    rope_theta_hw: float = 10_000.0

    def __post_init__(self) -> None:
        """Validate the rotation-pair and grouped-query head geometry."""
        if self.head_dim <= 0 or self.head_dim % 8:
            raise ValueError(f"head_dim must be a positive multiple of 8 for T/H/W RoPE, got {self.head_dim}")
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError(
                f"num_heads and num_kv_heads must be positive, got {self.num_heads} and {self.num_kv_heads}"
            )
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

    @property
    def rope_sections(self) -> tuple[int, int, int]:
        """Split T/H/W dimensions in a 2:1:1 ratio.

        Each axis contains rotation pairs and needs an even width, requiring
        head_dim to be divisible by eight.
        """
        return self.head_dim // 2, self.head_dim // 4, self.head_dim // 4


def patchify(video: torch.Tensor) -> torch.Tensor:
    """[B,3,T,H,W] -> [B, (T/4)(H/16)(W/16), 3072]; token order T,H,W, values (t,p,q,c)."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")
    pt, ph, pw = TUBELET
    batch, channels, frames, height, width = video.shape
    if frames % pt or height % ph or width % pw:
        raise ValueError(f"video {tuple(video.shape[2:])} must be a multiple of {TUBELET}")
    x = video.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
    return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(batch, -1, TUBE_DIM)


def unpatchify(tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    batch, count, dim = tokens.shape
    nt, nh, nw = grid
    pt, ph, pw = TUBELET
    if count != nt * nh * nw or dim != TUBE_DIM:
        raise ValueError(f"tokens {tuple(tokens.shape)} do not match grid {grid}")
    x = tokens.reshape(batch, nt, nh, nw, pt, ph, pw, 3)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch, 3, nt * pt, nh * ph, nw * pw)


def mrope_tables(arch: PixelUMMArchitecture, grid: tuple[int, int, int], device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [1, T*H*W, 1, head_dim] for a T/H/W-ordered grid, laid out [T | H | W]."""
    axes = [torch.arange(size, device=device) for size in grid]
    positions = torch.meshgrid(*axes, indexing="ij")
    thetas = (arch.rope_theta_t, arch.rope_theta_hw, arch.rope_theta_hw)
    cos_parts, sin_parts = [], []
    for position, dim, theta in zip(positions, arch.rope_sections, thetas):
        inv_freq = theta ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        angles = position.flatten().float()[:, None] * inv_freq[None]
        angles = torch.cat((angles, angles), dim=-1)
        cos_parts.append(angles.cos())
        sin_parts.append(angles.sin())
    return (
        torch.cat(cos_parts, dim=-1)[None, :, None, :],
        torch.cat(sin_parts, dim=-1)[None, :, None, :],
    )


def local_attention_layout(grid, tile, device):
    frames, height, width = grid
    tiles_h = math.ceil(height / tile)
    tiles_w = math.ceil(width / tile)
    if tiles_h <= 3 and tiles_w <= 3:
        return None
    axes = [torch.arange(size, device=device) for size in grid]
    time, row, column = (axis.flatten() for axis in torch.meshgrid(*axes, indexing="ij"))
    # Sort tokens tile-major: tile, then frame, then the row and column inside the tile.
    tile_index = (row // tile) * tiles_w + column // tile
    order = tile_index * frames + time
    order = order * tile + row % tile
    order = order * tile + column % tile
    permutation = torch.argsort(order)
    ranges = []
    offset = 0
    for row_index in range(tiles_h):
        for column_index in range(tiles_w):
            count = frames * min(tile, height - row_index * tile) * min(tile, width - column_index * tile)
            ranges.append((offset, offset + count))
            offset += count
    neighborhoods = []
    for row_index in range(tiles_h):
        for column_index in range(tiles_w):
            # Edge tiles shift their 3x3 neighborhood inward.
            center_h = min(max(row_index, 1), max(tiles_h - 2, 1))
            center_w = min(max(column_index, 1), max(tiles_w - 2, 1))
            neighbors = []
            for h in range(max(0, center_h - 1), min(tiles_h, center_h + 2)):
                for w in range(max(0, center_w - 1), min(tiles_w, center_w + 2)):
                    neighbors.append(ranges[h * tiles_w + w])
            neighborhoods.append((ranges[row_index * tiles_w + column_index], neighbors))
    return permutation, torch.argsort(permutation), neighborhoods


class PixelUMMRMSNorm(nn.Module):
    """Qwen3 RMSNorm for the split Q/K halves: normalize in fp32, cast back to the input dtype, then scale.

    Q/K enter in FP32 and stay FP32 through RoPE, which vLLM's RMSNorm (it
    rounds to the weight dtype) cannot express.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _norm_rope(part, cos, sin, norm_t, norm_hw, sections, dtype):
    temporal, spatial = part.float().chunk(2, dim=-1)
    normalized = torch.cat((norm_t(temporal), norm_hw(spatial)), dim=-1)
    return apply_mrope(normalized, cos, sin, sections).to(dtype)


def prepare_qk(tensor, cos, sin, norm_t, norm_hw, sections, dtype):
    """Split FP32 Q/K norm and MRoPE for ``[B, N, heads, D]``, rounded once to ``dtype``."""
    if dtype == tensor.dtype:
        fused = split_norm_rope(tensor, cos, sin, norm_t.weight, norm_hw.weight, norm_t.eps)
        if fused is not None:
            return fused
    # Bound FP32 intermediates while preserving each token's reduction order.
    batch, tokens, heads, dim = tensor.shape
    chunk = max(1, _QK_CHUNK_INPUT_BYTES // (batch * heads * dim * 4))
    if torch.is_grad_enabled() or tokens <= chunk:
        return _norm_rope(tensor, cos, sin, norm_t, norm_hw, sections, dtype)
    result = torch.empty_like(tensor, dtype=dtype)
    for first in range(0, tokens, chunk):
        last = min(first + chunk, tokens)
        part = tensor[:, first:last]
        part = _norm_rope(part, cos[:, first:last], sin[:, first:last], norm_t, norm_hw, sections, dtype)
        result[:, first:last].copy_(part)
    return result


class PixelUMMAttention(nn.Module):
    def __init__(self, arch: PixelUMMArchitecture, prefix: str = ""):
        super().__init__()
        self.arch = arch
        hidden, dim = arch.hidden_size, arch.head_dim
        # Qwen3 layout: one stacked QKV projection, heads sharded over tensor-parallel ranks.
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = arch.num_heads // tp_size
        self.num_kv_heads = max(1, arch.num_kv_heads // tp_size)
        self.qkv_proj = QKVParallelLinear(
            hidden, dim, arch.num_heads, arch.num_kv_heads, bias=False, prefix=f"{prefix}.qkv_proj"
        )
        self.o_proj = RowParallelLinear(arch.num_heads * dim, hidden, bias=False, prefix=f"{prefix}.o_proj")
        self.q_norm_t = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.q_norm_hw = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_t = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_hw = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
        self.attention = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=dim,
            softmax_scale=dim**-0.5,
            causal=False,
            role="hy_nomas.decoder",
            role_category="self",
            skip_sequence_parallel=True,
        )

    def forward(self, hidden_states, cos, sin, neighborhoods=None, packed=None) -> torch.Tensor:
        batch, count, _ = hidden_states.shape
        dim = self.arch.head_dim
        qkv, _ = self.qkv_proj(hidden_states)
        # Q/K/V are strided views of the stacked projection.
        query_size = self.num_heads * dim
        kv_size = self.num_kv_heads * dim
        query, key, value = qkv.split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(batch, count, self.num_heads, dim)
        key = key.view(batch, count, self.num_kv_heads, dim)
        value = value.view(batch, count, self.num_kv_heads, dim)
        sections = self.arch.rope_sections
        query = prepare_qk(query, cos, sin, self.q_norm_t, self.q_norm_hw, sections, value.dtype)
        key = prepare_qk(key, cos, sin, self.k_norm_t, self.k_norm_hw, sections, value.dtype)
        # Keep K/V compressed through neighborhood gathering. The shared backend
        # selects fused GQA where supported and owns the equal-head fallback.
        if packed is not None:
            # One varlen call reads every tile's neighborhood (decoder_attention.py).
            output = packed(self.attention, query, key, value)
        elif neighborhoods is None:
            output = self.attention(query, key, value)
        else:
            outputs = []
            for (start, end), neighbors in neighborhoods:
                local_key = torch.cat([key[:, first:last] for first, last in neighbors], dim=1)
                local_value = torch.cat([value[:, first:last] for first, last in neighbors], dim=1)
                outputs.append(self.attention(query[:, start:end].contiguous(), local_key, local_value))
            output = torch.cat(outputs, dim=1)
        output, _ = self.o_proj(output.reshape(batch, count, self.num_heads * dim))
        return output


class PixelUMMMLP(nn.Module):
    """Qwen3 MLP: one stacked gate/up projection, then the shared SiLU*mul kernel."""

    def __init__(self, arch: PixelUMMArchitecture, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            arch.hidden_size, [arch.intermediate_size] * 2, bias=False, prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(
            arch.intermediate_size, arch.hidden_size, bias=False, prefix=f"{prefix}.down_proj"
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class PixelUMMBlock(nn.Module):
    def __init__(self, arch: PixelUMMArchitecture, prefix: str = ""):
        super().__init__()
        self.input_norm = RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.attn = PixelUMMAttention(arch, prefix=f"{prefix}.attn")
        self.post_attn_norm = RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.mlp = PixelUMMMLP(arch, prefix=f"{prefix}.mlp")

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, neighborhoods=None, packed=None
    ) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin, neighborhoods, packed)
        return x + self.mlp(self.post_attn_norm(x))


class PixelUMMHead(nn.Module):
    """Normalize hidden states and project each tubelet to RGB pixels."""

    def __init__(self, hidden: int, prefix: str = ""):
        super().__init__()
        self.norm = RMSNorm(hidden, eps=1e-6)
        self.linear = ReplicatedLinear(hidden, TUBE_DIM, bias=True, prefix=f"{prefix}.linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear(self.norm(x))
        return x


class PixelUMMDecoderModel(nn.Module):
    """`forward(latent [B,32,L,h,w], xt [B,3,4L,32h,32w]) -> x0 prediction`."""

    def __init__(self, arch: PixelUMMArchitecture, *, local_attention_tile: int = 0, prefix: str = "decoder"):
        super().__init__()
        if local_attention_tile < 0:
            raise ValueError(f"local_attention_tile must be >= 0 tokens, got {local_attention_tile}")
        self.arch = arch
        self.local_attention_tile = int(local_attention_tile)
        hidden = arch.hidden_size
        self.embed = ReplicatedLinear(TUBE_DIM, hidden, bias=True, prefix=f"{prefix}.embed")
        self.cond_proj = ReplicatedLinear(
            LATENT_CHANNELS, LATENT_TOKEN_GRID**2 * hidden, bias=True, prefix=f"{prefix}.cond_proj"
        )
        self.layers = nn.ModuleList(
            PixelUMMBlock(arch, prefix=f"{prefix}.layers.{index}") for index in range(arch.num_layers)
        )
        self.final_norm = RMSNorm(hidden, eps=arch.rms_norm_eps)
        self.head = PixelUMMHead(hidden, prefix=f"{prefix}.head")

    def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"expected DCAE latent [B,{LATENT_CHANNELS},L,h,w], got {tuple(latent.shape)}")
        batch, _, frames, height, width = latent.shape
        pixel_shape = (batch, 3, frames * TUBELET[0], height * 32, width * 32)
        if tuple(xt.shape) != pixel_shape:
            raise ValueError(f"xt {tuple(xt.shape)} != expected {pixel_shape}")
        dtype = latent.dtype
        if dtype in (torch.bfloat16, torch.float16):
            context = torch.autocast(device_type=latent.device.type, dtype=dtype)
        else:
            context = nullcontext()
        grid = (frames, height * LATENT_TOKEN_GRID, width * LATENT_TOKEN_GRID)
        with context:
            tokens, _ = self.embed(patchify(xt.to(dtype)))
            condition, _ = self.cond_proj(latent.to(dtype).permute(0, 2, 3, 4, 1))
            condition = condition.view(
                batch, frames, height, width, LATENT_TOKEN_GRID, LATENT_TOKEN_GRID, self.arch.hidden_size
            )
            condition = condition.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, -1, self.arch.hidden_size)
            tokens = tokens + condition.to(tokens.device)
            cos, sin = mrope_tables(self.arch, grid, tokens.device)
            layout = None
            if self.local_attention_tile:
                layout = local_attention_layout(grid, self.local_attention_tile, tokens.device)
            neighborhoods = None
            packed = None
            if layout is not None:
                perm, inverse, neighborhoods = layout
                tokens = tokens[:, perm]
                cos = cos[:, perm]
                sin = sin[:, perm]
                # Every layer's attention shares one backend and one packed plan.
                packed = packed_local_attention(self.layers[0].attn.attention, neighborhoods, tokens)
            for layer in self.layers:
                tokens = layer(tokens, cos, sin, neighborhoods, packed)
            tokens = self.final_norm(tokens)
            if layout is not None:
                tokens = tokens[:, inverse.to(tokens.device)]
        # The source runs the head after leaving autocast (forward_features returns first).
        return unpatchify(self.head(tokens), grid)


def _frame_seed(seed: int, index: int) -> int:
    """Noise seed of one absolute latent frame, so overlapping windows share noise."""
    return (seed * 1_000_003 + index) % (2**64)


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    if window < 1 or stride < 1 or stride > window:
        raise ValueError(f"expected 1 <= stride <= window, got stride={stride}, window={window}")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    tail = length - window
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _slice_along(tensor: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]


def _merge_overlapping(parts: list[torch.Tensor], starts: list[int], *, dim: int) -> torch.Tensor:
    if len(parts) != len(starts) or not parts:
        raise ValueError("parts and starts must be non-empty and have equal length")
    result = parts[0]
    result_end = starts[0] + parts[0].shape[dim]
    for part, start in zip(parts[1:], starts[1:]):
        part_end = start + part.shape[dim]
        overlap = result_end - start
        if overlap < 0 or part_end <= result_end:
            raise ValueError("overlap windows must be ordered and extend the range")
        overlap = min(overlap, result.shape[dim], part.shape[dim])
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = 0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))
        shape = [1] * part.ndim
        shape[dim] = overlap
        weight = weight.view(shape).to(dtype=part.dtype)
        result_head = _slice_along(result, dim, 0, result.shape[dim] - overlap)
        result_tail = _slice_along(result, dim, result.shape[dim] - overlap, result.shape[dim])
        part_head = _slice_along(part, dim, 0, overlap)
        part_tail = _slice_along(part, dim, overlap, part.shape[dim])
        result = torch.cat([result_head, result_tail * (1.0 - weight) + part_head * weight, part_tail], dim=dim)
        result_end = part_end
    return result


class _TemporalBlender:
    """Incremental twin of `_merge_overlapping(parts, starts, dim=2)` into one buffer.

    Same arithmetic as the source's `_merge_into_buffer`: one preallocated clip buffer, each window cosine-blended in
    as soon as it is complete.
    """

    def __init__(self, total_frames: int):
        self.total = int(total_frames)
        self.out: torch.Tensor | None = None
        self.end = 0

    def add(self, part: torch.Tensor, start: int) -> None:
        length = part.shape[2]
        if self.out is None:
            self.out = part.new_empty((*part.shape[:2], self.total, *part.shape[3:]))
            self.out[:, :, :length] = part
            self.end = length
            return
        overlap = self.end - start
        if overlap < 0 or start + length <= self.end:
            raise ValueError(
                "overlap windows must be ordered and extend the covered range: "
                f"start={start}, result_end={self.end}, part_end={start + length}"
            )
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = 0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))
        weight = weight.view(1, 1, -1, 1, 1).to(dtype=part.dtype)
        out = self.out
        previous = out[:, :, start : self.end]
        out[:, :, start : self.end] = previous * (1.0 - weight) + part[:, :, :overlap] * weight
        out[:, :, self.end : start + length] = part[:, :, overlap:]
        self.end = start + length

    def result(self) -> torch.Tensor:
        if self.out is None or self.end != self.total:
            raise RuntimeError(f"temporal merge incomplete: covered {self.end} of {self.total}")
        return self.out


@dataclass(frozen=True)
class PixelUMMDecodeConfig:
    temporal_window_latents: int
    temporal_overlap_latents: int
    decoder_tile_pixels: int
    decoder_tile_overlap_pixels: int
    denoising_step_list: tuple[float, ...]
    num_steps: int
    seed: int
    scaling_factor: float = 1.0
    shift_factor: float | None = None


class PixelUMMDecoder(nn.Module):
    """
    PixelUMM DMD2 decoder for DCAE-V video latents.

    This decoder predicts RGB pixels from 32-channel latents with spatial compression 32 and temporal compression 4.
    Local attention covers three neighboring spatial tiles along each axis and the complete temporal window. The model
    preserves the released decoder checkpoint layout.

    Parameters:
        hidden_size (`int`, defaults to 4096):
            Transformer hidden dimension.
        intermediate_size (`int`, defaults to 12288):
            Gated feed-forward intermediate dimension.
        num_layers (`int`, defaults to 36):
            Number of transformer layers.
        num_heads (`int`, defaults to 32):
            Number of query heads.
        num_kv_heads (`int`, defaults to 8):
            Number of key and value heads.
        head_dim (`int`, defaults to 128):
            Dimension of each attention head.
        rms_norm_eps (`float`, defaults to 1e-6):
            RMS normalization epsilon.
        rope_theta_t (`float`, defaults to 1000000.0):
            Rotary embedding frequency base for the temporal axis.
        rope_theta_hw (`float`, defaults to 10000.0):
            Rotary embedding frequency base for both spatial axes.
        local_attention_tile_pixels (`int`, defaults to 128):
            Spatial attention tile size in pixels. Zero enables full attention.
        denoising_step_list (`tuple[float]`, defaults to `(0.999, 0.866, 0.634, 0.342)`):
            Distilled noise schedule, in decreasing order.
        num_steps (`int`, defaults to 1):
            Number of entries to use from the distilled schedule.
        seed (`int`, defaults to 42):
            Pixel noise seed used when no generator is supplied.
        temporal_window_latents (`int`, defaults to 10):
            Number of latent frames decoded in each temporal window.
        temporal_overlap_latents (`int`, defaults to 5):
            Overlap between temporal windows, in latent frames.
        decoder_tile_pixels (`int`, defaults to 0):
            Spatial decode tile size in pixels. Zero decodes the full spatial field.
        decoder_tile_overlap_pixels (`int`, defaults to 64):
            Overlap between spatial decode tiles, in pixels.
        tail_hold_latents (`int`, defaults to 1):
            Number of tail latents withheld by preview consumers when a later segment will correct the tail.
    """

    _no_split_modules = ["PixelUMMBlock"]
    _repeated_blocks = ["PixelUMMBlock"]
    _layerwise_offload_blocks_attrs = ["decoder.layers"]
    _skip_layerwise_casting_patterns = ["embed", "cond_proj", "norm", "head"]

    spatial_compression_ratio = 32
    time_compression_ratio = 4
    temporal_compression_ratio = 4
    scaling_factor = 1.0

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_layers: int = 36,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta_t: float = 1_000_000.0,
        rope_theta_hw: float = 10_000.0,
        local_attention_tile_pixels: int = 128,
        denoising_step_list: tuple[float, ...] = (0.999, 0.866, 0.634, 0.342),
        num_steps: int = 1,
        seed: int = 42,
        temporal_window_latents: int = 10,
        temporal_overlap_latents: int = 5,
        decoder_tile_pixels: int = 0,
        decoder_tile_overlap_pixels: int = 64,
        tail_hold_latents: int = 1,
    ):
        super().__init__()
        if local_attention_tile_pixels < 0 or local_attention_tile_pixels % 16:
            raise ValueError("local_attention_tile_pixels must be zero or a multiple of 16")
        if temporal_window_latents < 1 or not 0 <= temporal_overlap_latents < temporal_window_latents:
            raise ValueError("temporal overlap must be nonnegative and smaller than the window")
        if decoder_tile_pixels < 0 or decoder_tile_pixels % 32:
            raise ValueError("decoder_tile_pixels must be zero or a multiple of 32")
        if decoder_tile_overlap_pixels < 0 or decoder_tile_overlap_pixels % 32:
            raise ValueError("decoder_tile_overlap_pixels must be a nonnegative multiple of 32")
        if decoder_tile_pixels and decoder_tile_overlap_pixels >= decoder_tile_pixels:
            raise ValueError("decoder overlap must be smaller than the tile")
        if tail_hold_latents < 0:
            raise ValueError("tail_hold_latents must be nonnegative")
        if not denoising_step_list:
            raise ValueError("denoising_step_list must contain values in (0, 1]")
        for value in denoising_step_list:
            if not 0 < value <= 1:
                raise ValueError("denoising_step_list must contain values in (0, 1]")
        for first, second in zip(denoising_step_list, denoising_step_list[1:]):
            if first <= second:
                raise ValueError("denoising_step_list must be strictly decreasing")
        if not 1 <= num_steps <= len(denoising_step_list):
            raise ValueError("num_steps must be between 1 and the distilled schedule length")
        arch = PixelUMMArchitecture(
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta_t,
            rope_theta_hw,
        )
        self.decoder = PixelUMMDecoderModel(arch, local_attention_tile=local_attention_tile_pixels // 16)
        self.temporal_window_latents = temporal_window_latents
        self.temporal_overlap_latents = temporal_overlap_latents
        self.tail_hold_latents = tail_hold_latents
        self.config = PixelUMMDecodeConfig(
            temporal_window_latents,
            temporal_overlap_latents,
            decoder_tile_pixels,
            decoder_tile_overlap_pixels,
            tuple(denoising_step_list),
            num_steps,
            seed,
        )
        self._decode_group: dist.ProcessGroup | None = None
        self._decode_error_status: torch.Tensor | None = None
        self._decode_output_rank = 0

    def configure_distributed_decode(self, process_group: dist.ProcessGroup, *, output_rank: int = 0) -> None:
        """Configure collective decoding on the supplied group."""
        if not dist.is_initialized():
            raise RuntimeError("PixelUMM window decoding requires torch.distributed")
        if process_group is None:
            raise ValueError("PixelUMM window decoding requires an explicit process group")
        size = dist.get_world_size(process_group)
        if size <= 1:
            raise ValueError("PixelUMM window decoding requires more than one rank")
        if not 0 <= output_rank < size:
            raise ValueError(f"PixelUMM output rank {output_rank} is outside group size {size}")
        # Keep the status on the decode device even when model weights are offloaded.
        self._decode_error_status = torch.empty(1, dtype=torch.int32, device=self.device)
        self._decode_group = process_group
        self._decode_output_rank = output_rank

    def _is_output_rank(self) -> bool:
        return self._decode_group is None or dist.get_rank(self._decode_group) == self._decode_output_rank

    @contextmanager
    def _decode_guard(self, operation: str) -> Iterator[None]:
        if self._decode_group is None:
            yield
            return
        error = None
        try:
            yield
        except Exception as exc:
            error = exc
        failed = self._decode_error_status
        if failed is None:
            raise RuntimeError("PixelUMM distributed error status is not initialized")
        failed.fill_(int(error is not None))
        dist.all_reduce(failed, group=self._decode_group)
        if error is not None:
            raise error
        if failed.item():
            raise RuntimeError(f"PixelUMM {operation} failed on another decode rank")

    def _validate_decode_metadata(self, values: Sequence[int]) -> None:
        if self._decode_group is None:
            return
        with self._decode_guard("metadata preparation"):
            length = torch.tensor([len(values)], dtype=torch.int64, device=self.device)
            lengths = [torch.empty_like(length) for _ in range(dist.get_world_size(self._decode_group))]
        dist.all_gather(lengths, length, group=self._decode_group)
        if any(not torch.equal(length, other) for other in lengths):
            raise ValueError("PixelUMM distributed decode requires identical metadata")
        with self._decode_guard("metadata allocation"):
            metadata = torch.tensor(values, dtype=torch.int64, device=self.device)
            peers = [torch.empty_like(metadata) for _ in lengths]
        dist.all_gather(peers, metadata, group=self._decode_group)
        if any(not torch.equal(metadata, other) for other in peers):
            raise ValueError("PixelUMM distributed decode requires identical metadata")

    def _validate_decode_latent(self, latent: torch.Tensor) -> None:
        if self._decode_group is None:
            return
        with self._decode_guard("latent validation"):
            dtype = str(latent.dtype).encode("ascii")
            decoder_dtype = str(self.dtype).encode("ascii")
            signature = (
                self._decode_output_rank,
                *latent.shape,
                len(dtype),
                *dtype,
                len(decoder_dtype),
                *decoder_dtype,
            )
        self._validate_decode_metadata(signature)
        with self._decode_guard("latent allocation"):
            local = latent.contiguous().to(device=self.device)
            # Broadcast uses an in-place buffer, separate from the caller's latent.
            reference = local.clone() if self._is_output_rank() else torch.empty_like(local)
        source = dist.get_global_rank(self._decode_group, self._decode_output_rank)
        dist.broadcast(reference, src=source, group=self._decode_group)
        with self._decode_guard("latent comparison"):
            equal = torch.tensor(int(torch.equal(local, reference)), dtype=torch.int32, device=self.device)
        dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=self._decode_group)
        if not equal.item():
            raise ValueError("PixelUMM distributed decode requires identical latent values")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint tensors, stacking q/k/v and gate/up projections like vLLM's Qwen3."""
        stacked_params_mapping = [
            # (parameter, checkpoint, shard id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        parameters = dict(self.named_parameters())
        # A stacked parameter is complete only once every checkpoint shard has loaded,
        # so completeness is checked by checkpoint name.
        expected = set()
        for name in parameters:
            stacked = False
            for parameter_name, checkpoint_name, _ in stacked_params_mapping:
                if parameter_name in name:
                    expected.add(name.replace(parameter_name, checkpoint_name))
                    stacked = True
            if not stacked:
                expected.add(name)
        loaded = set()
        loaded_checkpoint = set()
        for checkpoint_key, tensor in weights:
            name = checkpoint_key
            for parameter_name, checkpoint_name, shard_id in stacked_params_mapping:
                if checkpoint_name not in name:
                    continue
                name = name.replace(checkpoint_name, parameter_name)
                parameter = parameters[name]
                parameter.weight_loader(parameter, tensor, shard_id)
                break
            else:
                parameter = parameters[name]
                weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
                weight_loader(parameter, tensor)
            loaded.add(name)
            loaded_checkpoint.add(checkpoint_key)
        missing = expected - loaded_checkpoint
        if missing:
            raise ValueError(f"PixelUMM decoder checkpoint is missing {len(missing)} tensors: {sorted(missing)[:10]}")
        return loaded

    @property
    def device(self) -> torch.device:
        return self.decoder.embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.embed.weight.dtype

    def forward(
        self, latent: torch.Tensor, sample: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | tuple[torch.Tensor]:
        """
        Predict a clean RGB video from a noisy video and its latent conditioning.

        Args:
            latent (`torch.Tensor`):
                Conditioning with shape `(batch, 32, frames, height, width)`.
            sample (`torch.Tensor`):
                Noisy RGB pixels with shape `(batch, 3, frames * 4, height * 32, width * 32)`.
            return_dict (`bool`, defaults to `True`):
                Return a [`~models.autoencoders.vae.DecoderOutput`] instead of a one-element tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                Predicted clean pixels with the same shape as `sample`.
        """
        output = self.decoder(latent, sample)
        if return_dict:
            return DecoderOutput(sample=output)
        return (output,)

    def _make_noise(self, latent, start, end, seed):
        batch, _, _, height, width = latent.shape
        frames = []
        for index in range(start, end):
            generator = torch.Generator(device=latent.device).manual_seed(_frame_seed(seed, index))
            noise = torch.randn(
                (batch, 3, 4, height * 32, width * 32), generator=generator, device=latent.device, dtype=torch.float32
            )
            frames.append(noise)
        return torch.cat(frames, dim=2)

    def _noise_tile(self, latent, seed, frame_offset=0):
        cached_range = None
        cached_noise = None

        def source(first, last, row, column, height, width):
            nonlocal cached_range, cached_noise
            if cached_range != (first, last):
                cached_noise = self._make_noise(latent, first + frame_offset, last + frame_offset, seed)
                cached_range = (first, last)
            return cached_noise[..., row * 32 : (row + height) * 32, column * 32 : (column + width) * 32]

        return source

    def _tile_grid(self, height: int, width: int) -> tuple[list[int], list[int], int, int]:
        cells = self.config.decoder_tile_pixels // 32
        if cells <= 0 or (height <= cells and width <= cells):
            return [0], [0], height, width
        spatial_stride = cells - self.config.decoder_tile_overlap_pixels // 32
        tile_h, tile_w = min(cells, height), min(cells, width)
        rows = _window_starts(height, tile_h, min(spatial_stride, tile_h))
        columns = _window_starts(width, tile_w, min(spatial_stride, tile_w))
        return rows, columns, tile_h, tile_w

    @staticmethod
    def _merge_tiles(tiles: list[torch.Tensor], rows: list[int], columns: list[int]) -> torch.Tensor:
        if len(tiles) == 1:
            return tiles[0]
        column_pixels = [column * 32 for column in columns]
        row_outputs = [
            _merge_overlapping(tiles[row * len(columns) : (row + 1) * len(columns)], column_pixels, dim=-1)
            for row in range(len(rows))
        ]
        return _merge_overlapping(row_outputs, [row * 32 for row in rows], dim=-2)

    def _predict_x0(self, latent, xt_source, seed, t_starts=None, pad_short=False, *, all_ranks=False):
        with self._decode_guard("window preparation"):
            frames, height, width = latent.shape[2:]
            window = self.config.temporal_window_latents
            stride = window - self.config.temporal_overlap_latents
            if t_starts is None:
                t_starts = _window_starts(frames, window, stride)
            t_starts = list(t_starts)
            length = min(window, frames)
            rows, columns, tile_h, tile_w = self._tile_grid(height, width)
            pad_noise = None
            if pad_short and frames >= window and t_starts[-1] + window > frames:
                variance, mean = torch.var_mean(latent.detach().float(), dim=(2, 3, 4), correction=0, keepdim=True)
                generator = torch.Generator(device=latent.device).manual_seed(_frame_seed(seed, frames - 1))
                noise = torch.randn(
                    (latent.shape[0], latent.shape[1], window, height, width),
                    generator=generator,
                    device=latent.device,
                    dtype=torch.float32,
                )
                pad_noise = mean + variance.sqrt() * noise
            row_pixels = [row * 32 for row in rows]
            column_pixels = [column * 32 for column in columns]
            blender = _TemporalBlender((t_starts[-1] + length - t_starts[0]) * 4)
        if self._decode_group is not None:
            signed_seed = (seed + (1 << 63)) % (1 << 64) - (1 << 63)
            self._validate_decode_metadata(
                (
                    int(all_ranks),
                    int(pad_short),
                    self._decode_output_rank,
                    signed_seed,
                    self.config.num_steps,
                    frames,
                    height,
                    width,
                    window,
                    stride,
                    length,
                    tile_h,
                    tile_w,
                    len(t_starts),
                    len(rows),
                    len(columns),
                    *t_starts,
                    *rows,
                    *columns,
                )
            )
            with self._decode_guard("task preparation"):
                spatial = [(row, column) for row in rows for column in columns]
                tasks = [(index, row, column) for index in range(len(t_starts)) for row, column in spatial]
            size = dist.get_world_size(self._decode_group)
            rank = dist.get_rank(self._decode_group)
            receives = all_ranks or self._is_output_rank()
            destination = dist.get_global_rank(self._decode_group, self._decode_output_rank)
            idle_shape = (latent.shape[0], 3, length * 4, tile_h * 32, tile_w * 32)
            pending: dict[int, list[torch.Tensor | None]] = {}
            next_window = 0
            for first in range(0, len(tasks), size):
                index = first + rank
                with self._decode_guard("window computation"):
                    if index < len(tasks):
                        temporal, row, column = tasks[index]
                        start = t_starts[temporal]
                        real = latent[:, :, start : start + length, row : row + tile_h, column : column + tile_w]
                        if pad_noise is not None and real.shape[2] < length:
                            fill = length - real.shape[2]
                            offset = (start // stride) % (window - fill + 1)
                            tile = pad_noise[:, :, offset : offset + fill, row : row + tile_h, column : column + tile_w]
                            real = torch.cat([real, tile.to(real.dtype)], dim=2)
                        noise = xt_source(start, start + length, row, column, tile_h, tile_w)
                        local = self.decoder(real.contiguous(), noise.contiguous()).contiguous()
                    else:
                        local = torch.empty(idle_shape, device=latent.device, dtype=self.dtype)
                    if tuple(local.shape) != idle_shape or local.dtype != self.dtype or local.device != latent.device:
                        raise RuntimeError("PixelUMM window output does not match the distributed tensor contract")
                    gathered = [torch.empty_like(local) for _ in range(size)] if receives else None
                if all_ranks:
                    dist.all_gather(gathered, local, group=self._decode_group)
                else:
                    dist.gather(local, gather_list=gathered, dst=destination, group=self._decode_group)
                with self._decode_guard("window merge"):
                    if receives:
                        assert gathered is not None
                        for source_rank, output in enumerate(gathered):
                            task_index = first + source_rank
                            if task_index >= len(tasks):
                                break
                            temporal, tile = divmod(task_index, len(spatial))
                            pending.setdefault(temporal, [None] * len(spatial))[tile] = output
                        del output
                        while next_window in pending and all(tile is not None for tile in pending[next_window]):
                            tiles = pending.pop(next_window)
                            blender.add(
                                self._merge_tiles(tiles, rows, columns), (t_starts[next_window] - t_starts[0]) * 4
                            )
                            del tiles
                            next_window += 1
                del gathered, local
            result = None
            with self._decode_guard("window finalization"):
                if receives:
                    if pending or next_window != len(t_starts):
                        raise RuntimeError("PixelUMM window decode left incomplete windows")
                    result = blender.result()
            return result
        for first in t_starts:
            row_outputs = []
            for row in rows:
                tile_outputs = []
                for column in columns:
                    real = latent[:, :, first : first + length, row : row + tile_h, column : column + tile_w]
                    if pad_noise is not None and real.shape[2] < length:
                        fill = length - real.shape[2]
                        offset = (first // stride) % (window - fill + 1)
                        tile = pad_noise[:, :, offset : offset + fill, row : row + tile_h, column : column + tile_w]
                        real = torch.cat([real, tile.to(real.dtype)], dim=2)
                    noise = xt_source(first, first + length, row, column, tile_h, tile_w)
                    tile_outputs.append(self.decoder(real.contiguous(), noise.contiguous()))
                row_outputs.append(_merge_overlapping(tile_outputs, column_pixels, dim=-1))
            decoded = _merge_overlapping(row_outputs, row_pixels, dim=-2)
            blender.add(decoded, (first - t_starts[0]) * 4)
        return blender.result()

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        pixel_frames: int | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        """
        Decode a latent clip with overlapping temporal windows and optional spatial tiles.

        Args:
            latents (`torch.Tensor`):
                Latent video with shape `(batch, 32, frames, height, width)`.
            generator (`torch.Generator`, *optional*):
                Its initial seed selects the pixel noise stream. Each absolute latent-frame index has its own seed, so
                overlapping windows receive identical noise. The generator state is not advanced.
            return_dict (`bool`, defaults to `True`):
                Return a [`~models.autoencoders.vae.DecoderOutput`] instead of a one-element tuple.
            pixel_frames (`int`, *optional*):
                Keep this many leading decoded frames; otherwise return `frames * 4` frames.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                RGB pixels with shape `(batch, 3, pixel_frames, height * 32, width * 32)`.
        """
        with self._decode_guard("offline input validation"):
            if latents.ndim != 5 or latents.shape[1] != 32 or min(latents.shape[2:]) < 1:
                raise ValueError("latents must have nonempty shape (batch, 32, frames, height, width)")
            seed = self._seed(generator)
            steps = self.config.denoising_step_list[: self.config.num_steps]
            signed_seed = (seed + (1 << 63)) % (1 << 64) - (1 << 63)
            target_frames = -1 if pixel_frames is None else int(pixel_frames)
            schedule = (
                tuple(struct.unpack("!q", struct.pack("!d", float(step)))[0] for step in steps)
                if self._decode_group is not None
                else ()
            )
        self._validate_decode_metadata((self.config.num_steps, signed_seed, target_frames, *schedule))
        self._validate_decode_latent(latents)
        if len(steps) == 1:
            video = self._predict_x0(latents, self._noise_tile(latents, seed), seed)
        else:
            with self._decode_guard("multistep state preparation"):
                state = self._make_noise(latents, 0, latents.shape[2], seed)

            def source(first, last, row, column, height, width):
                return state[
                    :, :, first * 4 : last * 4, row * 32 : (row + height) * 32, column * 32 : (column + width) * 32
                ]

            for index, current in enumerate(steps):
                video = self._predict_x0(latents, source, seed, all_ranks=index + 1 < len(steps))
                if index + 1 < len(steps):
                    with self._decode_guard("multistep state update"):
                        assert video is not None
                        ratio = steps[index + 1] / current
                        for offset in range(0, state.shape[2], 40):
                            end = offset + 40
                            anchor = video[:, :, offset:end].to(device=state.device, dtype=torch.float32)
                            state[:, :, offset:end] = anchor + ratio * (state[:, :, offset:end] - anchor)
        with self._decode_guard("offline output preparation"):
            if video is None:
                video = latents.new_empty((0,))
            elif pixel_frames is not None:
                if not 1 <= pixel_frames <= video.shape[2]:
                    raise ValueError("pixel_frames must be positive and no larger than the decoded frame count")
                video = video[:, :, :pixel_frames]
        if return_dict:
            return DecoderOutput(sample=video)
        return (video,)

    def _seed(self, generator: torch.Generator | None) -> int:
        if generator is None:
            return self.config.seed
        return generator.initial_seed()

    @torch.no_grad()
    def _decode_windows(
        self, latent: torch.Tensor, starts: list[int], seed: int, pad_short: bool = False, frame_offset: int = 0
    ) -> torch.Tensor | None:
        with self._decode_guard("stream window preparation"):
            latent = latent.to(device=self.device, dtype=self.dtype)
            xt_source = self._noise_tile(latent, seed, frame_offset)
        return self._predict_x0(latent, xt_source, seed, t_starts=starts, pad_short=pad_short)

    def create_streaming_decode_session(
        self,
        *,
        to_cpu: bool = True,
        retain_history: bool = False,
        generator: torch.Generator | None = None,
        rolling: bool = True,
    ) -> "PixelUMMStreamingDecodeSession":
        """
        Create session state for stable append-only decoding.

        Streaming requires one decoder step. Only future-stable pixels are returned
        by push(); finish() emits the withheld final tail. The optional generator
        selects the same absolute-frame noise stream as decode().
        The default rolling mode, which AR sessions use, discards completed prefixes
        and requires retain_history=False.
        """
        with self._decode_guard("stream session initialization"):
            if self.config.num_steps != 1:
                raise ValueError("Streaming PixelUMM decoding requires num_steps=1")
            seed = self._seed(generator)
            session = PixelUMMStreamingDecodeSession(
                self, to_cpu=to_cpu, retain_history=retain_history, seed=seed, rolling=rolling
            )
        return session


class PixelUMMStreamingDecodeSession:
    """Session-owned stable-prefix decoder; never publishes provisional pixels."""

    def __init__(
        self, model, *, to_cpu: bool = True, retain_history: bool = True, seed: int = 42, rolling: bool = False
    ):
        self.model = model
        self.to_cpu = bool(to_cpu)
        self.retain_history = bool(retain_history)
        if rolling and retain_history:
            raise ValueError("rolling decoding requires retain_history=False")
        self.rolling = bool(rolling)
        self.buffer_latent_start = 0
        self.segment_raw_start = 0
        self.segment_visible_start = 0
        self.window = model.temporal_window_latents
        self.stride = self.window - model.temporal_overlap_latents
        self.temporal_ratio = model.time_compression_ratio
        self.buffer: torch.Tensor | None = None
        self.available_latent_frames = 0
        self.available_pixel_frames = 0
        self.next_window_start = 0
        self.pending: torch.Tensor | None = None
        self.pending_pixel_start = 0
        self.committed: list[tuple[int, torch.Tensor]] = []
        self.committed_raw_pixel_frames = 0
        self.committed_pixel_frames = 0
        self.push_latent_frames: list[int] = []
        self.push_pixel_frames: list[int] = []
        self.finished = False
        self.seed = seed
        self.final_segment = True

    def _validate_parallel_state(self, operation: int, pixel_frames: int) -> None:
        if self.model._decode_group is None:
            return
        with self.model._decode_guard("stream state preparation"):
            shape = () if self.buffer is None else tuple(self.buffer.shape)
            signed_seed = (self.seed + (1 << 63)) % (1 << 64) - (1 << 63)
            values = (
                operation,
                pixel_frames,
                int(self.final_segment),
                int(self.finished),
                int(self.to_cpu),
                int(self.retain_history),
                int(self.rolling),
                signed_seed,
                self.model._decode_output_rank,
                self.window,
                self.stride,
                self.temporal_ratio,
                self.available_latent_frames,
                self.available_pixel_frames,
                self.next_window_start,
                self.committed_raw_pixel_frames,
                self.committed_pixel_frames,
                self.buffer_latent_start,
                self.segment_raw_start,
                self.segment_visible_start,
                len(shape),
                len(self.push_latent_frames),
                len(self.push_pixel_frames),
                *shape,
                *self.push_latent_frames,
                *self.push_pixel_frames,
            )
        self.model._validate_decode_metadata(values)

    def _append_decoded_windows(self, video: torch.Tensor, *, first_window_start: int) -> None:
        clip_start = first_window_start * self.temporal_ratio
        if self.pending is None:
            self.pending = video
            self.pending_pixel_start = clip_start
            return
        relative_start = clip_start - self.pending_pixel_start
        self.pending = _merge_overlapping([self.pending, video], [0, relative_start], dim=2)

    @property
    def available_raw_pixel_frames(self) -> int:
        return self.available_latent_frames * self.temporal_ratio

    def _segment_raw_visible_spans(self):
        raw_start = self.segment_raw_start
        visible_start = self.segment_visible_start
        for latent_frames, pixel_frames in zip(self.push_latent_frames, self.push_pixel_frames):
            raw_end = raw_start + latent_frames * self.temporal_ratio
            visible_end = visible_start + pixel_frames
            yield (raw_start, raw_start + pixel_frames, raw_end, visible_start, visible_end)
            raw_start = raw_end
            visible_start = visible_end

    def _visible_frames_before_raw(self, raw_end: int) -> int:
        raw_end = max(0, min(int(raw_end), self.available_raw_pixel_frames))
        visible = self.segment_visible_start
        for raw_start, raw_visible_end, _, _, visible_end in self._segment_raw_visible_spans():
            if raw_end <= raw_start:
                break
            if raw_end < raw_visible_end:
                return visible + raw_end - raw_start
            visible = visible_end
        return visible

    def _visible_slice_from_raw(self, video: torch.Tensor, *, raw_start: int, raw_end: int) -> torch.Tensor:
        """Drop per-segment decoder padding from one raw-pixel tensor."""
        raw_start = int(raw_start)
        raw_end = int(raw_end)
        expected = raw_end - raw_start
        if expected < 0 or int(video.shape[2]) != expected:
            raise RuntimeError(
                f"PixelUMM raw preview range mismatch: range=[{raw_start}, {raw_end}), video={video.shape[2]}"
            )
        parts = []
        for segment_raw_start, segment_visible_raw_end, _, _, _ in self._segment_raw_visible_spans():
            overlap_start = max(raw_start, segment_raw_start)
            overlap_end = min(raw_end, segment_visible_raw_end)
            if overlap_end <= overlap_start:
                continue
            parts.append(video[:, :, overlap_start - raw_start : overlap_end - raw_start])
        if parts:
            return torch.cat(parts, dim=2)
        return video[:, :, :0]

    def _provisional_window_starts(self) -> list[int]:
        if self.model.tail_hold_latents and not self.final_segment:
            length = self.available_latent_frames
            if length <= self.window:
                return [0]
            return list(range(0, length, self.stride))
        if self.rolling and self.available_latent_frames > self.window:
            committed_latent = self.committed_raw_pixel_frames // self.temporal_ratio
            first = max(0, (committed_latent - self.window) // self.stride + 1)
            tail = self.available_latent_frames - self.window
            starts = list(range(first * self.stride, tail + 1, self.stride))
            if not starts or starts[-1] != tail:
                starts.append(tail)
            return starts
        return _window_starts(self.available_latent_frames, self.window, self.stride)

    def _trim_completed_prefix(self) -> None:
        """Release completed latent storage and segment mappings without resetting absolute positions."""
        if not self.rolling:
            return
        assert self.buffer is not None
        stable_latent = self.committed_raw_pixel_frames // self.temporal_ratio
        keep = max(0, (stable_latent - self.window) // self.stride + 1) * self.stride
        keep = min(keep, self.next_window_start)
        if keep > self.buffer_latent_start:
            self.buffer = self.buffer[:, :, keep - self.buffer_latent_start :].clone()
            self.buffer_latent_start = keep
        count = 0
        for latent_frames, pixel_frames in zip(self.push_latent_frames, self.push_pixel_frames):
            raw_end = self.segment_raw_start + latent_frames * self.temporal_ratio
            if raw_end > self.committed_raw_pixel_frames:
                break
            self.segment_raw_start = raw_end
            self.segment_visible_start += pixel_frames
            count += 1
        if count:
            del self.push_latent_frames[:count]
            del self.push_pixel_frames[:count]

    def _decode_provisional_raw(self) -> torch.Tensor | None:
        """Decode the exact current tail from the raw stable boundary onward."""
        with self.model._decode_guard("stream tail planning"):
            assert self.buffer is not None
            raw_start = self.committed_raw_pixel_frames
            raw_end = self.available_raw_pixel_frames
            selected_starts = None
            result = None
            regular_end = self.next_window_start - self.stride + self.window if self.next_window_start > 0 else 0
            if raw_end <= raw_start:
                if self.model._is_output_rank():
                    result = self.pending[:, :, :0] if self.pending is not None else self.buffer.new_empty((0,))
            elif raw_end <= regular_end * self.temporal_ratio:
                if self.model._is_output_rank():
                    if self.pending is None:
                        raise RuntimeError("PixelUMM session is missing pending regular-window output")
                    pending_end = self.pending_pixel_start + int(self.pending.shape[2])
                    if not self.pending_pixel_start <= raw_start <= raw_end <= pending_end:
                        raise RuntimeError(
                            "PixelUMM pending output does not cover the regular decoded range "
                            f"[{raw_start}, {raw_end}); "
                            f"pending=[{self.pending_pixel_start}, {pending_end})"
                        )
                    result = self.pending[
                        :, :, raw_start - self.pending_pixel_start : raw_end - self.pending_pixel_start
                    ].contiguous()
            else:
                starts = self._provisional_window_starts()
                committed_latent = raw_start // self.temporal_ratio
                first_index = next(
                    (index for index, start in enumerate(starts) if start + self.window > committed_latent),
                    len(starts) - 1,
                )
                selected_starts = starts[first_index:]
        if selected_starts is None:
            return result
        decoded = self._decode_windows(selected_starts)
        with self.model._decode_guard("stream tail slicing"):
            if self.model._is_output_rank():
                assert decoded is not None
                decoded_raw_start = selected_starts[0] * self.temporal_ratio
                local_start, local_end = raw_start - decoded_raw_start, raw_end - decoded_raw_start
                if not 0 <= local_start <= local_end <= decoded.shape[2]:
                    raise RuntimeError(
                        "PixelUMM provisional decode does not cover the requested raw range: "
                        f"decoded_start={decoded_raw_start}, requested=[{raw_start}, {raw_end}), "
                        f"decoded_frames={decoded.shape[2]}"
                    )
                result = decoded[:, :, local_start:local_end].contiguous()
        return result

    def _commit_until(self, stable_raw_until: int) -> torch.Tensor | None:
        stable_raw_until = int(stable_raw_until)
        if stable_raw_until <= self.committed_raw_pixel_frames:
            return None
        if not self.model._is_output_rank():
            self.committed_raw_pixel_frames = stable_raw_until
            self.committed_pixel_frames = self._visible_frames_before_raw(stable_raw_until)
            return None
        if self.pending is None:
            raise RuntimeError("PixelUMM session has no decoded pixels to commit")
        visible_start = self.committed_pixel_frames
        stable_local_end = stable_raw_until - self.pending_pixel_start
        if not 0 < stable_local_end <= self.pending.shape[2]:
            raise RuntimeError(
                f"Invalid PixelUMM stable-prefix boundary: local_end={stable_local_end}, "
                f"pending={self.pending.shape[2]}"
            )
        stable_raw = self.pending[:, :, :stable_local_end].contiguous()
        stable = self._visible_slice_from_raw(stable_raw, raw_start=self.pending_pixel_start, raw_end=stable_raw_until)
        if self.to_cpu:
            stable = stable.cpu()
        if self.retain_history and stable.shape[2] > 0:
            self.committed.append((visible_start, stable))
        self.pending = self.pending[:, :, stable_local_end:].contiguous()
        self.pending_pixel_start = stable_raw_until
        self.committed_raw_pixel_frames = stable_raw_until
        self.committed_pixel_frames = self._visible_frames_before_raw(stable_raw_until)
        return stable

    def _decode_ready_regular_windows(self) -> torch.Tensor | None:
        with self.model._decode_guard("stream window planning"):
            starts = []
            while self.next_window_start + self.window <= self.available_latent_frames:
                starts.append(self.next_window_start)
                self.next_window_start += self.stride
        if not starts:
            return None
        decoded = self._decode_windows(starts)
        with self.model._decode_guard("stream stable-prefix commit"):
            if self.model._is_output_rank():
                assert decoded is not None
                self._append_decoded_windows(decoded, first_window_start=starts[0])
            tail_start = self.available_latent_frames - self.window
            if tail_start >= 0 and tail_start % self.stride == 0:
                stable_latent_frames = tail_start + 1
            else:
                stable_latent_frames = max(0, tail_start)
            stable = self._commit_until(stable_latent_frames * self.temporal_ratio)
        return stable

    @property
    def stable_pixel_frames(self) -> int:
        return self.committed_pixel_frames

    @property
    def withheld_pixel_frames(self) -> int:
        return self.available_pixel_frames - self.committed_pixel_frames

    def slice_pixels(self, start: int, end: int) -> torch.Tensor:
        """Read a future-stable pixel range from retained history."""
        if not self.retain_history:
            raise RuntimeError("PixelUMM streaming history retention is disabled for this session.")
        if not 0 <= start < end <= self.committed_pixel_frames:
            raise ValueError(
                f"Requested PixelUMM pixels are not stable yet: range=[{start}, {end}), "
                f"stable={self.committed_pixel_frames}."
            )
        if not self.model._is_output_rank():
            assert self.buffer is not None
            return self.buffer.new_empty((0,))
        selected = []
        cursor = start
        for part_start, part in self.committed:
            part_end = part_start + int(part.shape[2])
            overlap_start = max(start, part_start)
            overlap_end = min(end, part_end)
            if overlap_end <= overlap_start:
                continue
            if overlap_start != cursor:
                raise RuntimeError(f"PixelUMM streaming output has a pixel gap: cursor={cursor}, next={overlap_start}.")
            selected.append(part[:, :, overlap_start - part_start : overlap_end - part_start])
            cursor = overlap_end
        if cursor != end or not selected:
            raise RuntimeError(
                f"PixelUMM streaming output does not cover requested range: [{start}, {end}), covered_to={cursor}."
            )
        return torch.cat(selected, dim=2)

    @torch.no_grad()
    def push(self, latent: torch.Tensor, *, pixel_frames: int | None = None) -> torch.Tensor:
        """Append one arbitrary-length segment and return newly stable pixels."""
        with self.model._decode_guard("stream input validation"):
            if self.finished:
                raise RuntimeError("Cannot push PixelUMM latents after finish().")
            if latent.ndim != 5 or latent.shape[2] <= 0:
                raise ValueError(f"expected non-empty [B,C,T,H,W], got {tuple(latent.shape)}")
            latent_frames = int(latent.shape[2])
            segment_pixel_frames = latent_frames * self.temporal_ratio if pixel_frames is None else int(pixel_frames)
            if segment_pixel_frames <= 0:
                raise ValueError("pixel_frames must be positive")
            min_pixel_frames = (latent_frames - 1) * self.temporal_ratio + 1
            max_pixel_frames = latent_frames * self.temporal_ratio
            if not min_pixel_frames <= segment_pixel_frames <= max_pixel_frames:
                raise ValueError(
                    "PixelUMM streaming segment pixel length is inconsistent with its latent length: "
                    f"latent_frames={latent_frames}, pixel_frames={segment_pixel_frames}, "
                    f"expected_range=[{min_pixel_frames}, {max_pixel_frames}]."
                )
        self._validate_parallel_state(0, segment_pixel_frames)
        self.model._validate_decode_latent(latent)
        with self.model._decode_guard("stream buffer append"):
            previous_stable = self.committed_pixel_frames
            if self.buffer is not None:
                self.buffer = torch.cat([self.buffer, latent], dim=2)
            elif self.rolling:
                # Rolling sessions later slice the buffer; own the storage.
                self.buffer = latent.clone()
            else:
                self.buffer = latent.contiguous()
            self.available_latent_frames += latent_frames
            self.available_pixel_frames += segment_pixel_frames
            self.push_latent_frames.append(latent_frames)
            self.push_pixel_frames.append(segment_pixel_frames)
        stable = self._decode_ready_regular_windows()
        with self.model._decode_guard("stream push finalization"):
            self._trim_completed_prefix()
            if stable is None or self.committed_pixel_frames <= previous_stable:
                assert self.buffer is not None
                stable = self.buffer.new_empty((0,))
        return stable

    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        """Publish the withheld tail and close the append-only timeline."""
        with self.model._decode_guard("stream finish validation"):
            if self.rolling and not self.final_segment:
                raise ValueError("rolling finish requires final_segment=True on the closing segment")
            if self.finished:
                raise RuntimeError("PixelUMM streaming decode is already finished.")
            if self.buffer is None:
                raise RuntimeError("Cannot finish an empty PixelUMM streaming session.")
        self._validate_parallel_state(1, -1)
        previous_stable = self.committed_pixel_frames
        provisional_raw = self._decode_provisional_raw()
        with self.model._decode_guard("stream finish commit"):
            stable = None
            if self.model._is_output_rank():
                assert provisional_raw is not None
                stable = self._visible_slice_from_raw(
                    provisional_raw, raw_start=self.committed_raw_pixel_frames, raw_end=self.available_raw_pixel_frames
                )
                if self.to_cpu:
                    stable = stable.cpu()
                if self.retain_history and stable.shape[2] > 0:
                    self.committed.append((previous_stable, stable))
            if self.pending is not None:
                self.pending = self.pending[:, :, :0]
            self.pending_pixel_start = self.available_raw_pixel_frames
            self.committed_raw_pixel_frames = self.available_raw_pixel_frames
            self.committed_pixel_frames = self.available_pixel_frames
            self.finished = True
            if self.rolling:
                assert self.buffer is not None
                self.buffer = self.buffer[:, :, :0].clone()
                self.pending = None
                self.push_latent_frames.clear()
                self.push_pixel_frames.clear()
            if stable is None or self.committed_pixel_frames <= previous_stable:
                assert self.buffer is not None
                stable = self.buffer.new_empty((0,))
        return stable

    def full_video(self) -> torch.Tensor:
        """Return the finalized full video from retained history."""
        if not self.finished:
            raise RuntimeError("PixelUMM streaming decode is not finished")
        if not self.retain_history:
            raise RuntimeError("PixelUMM streaming history retention is disabled for this session.")
        return self.slice_pixels(0, self.available_pixel_frames)

    def _decode_windows(self, starts):
        with self.model._decode_guard("stream window input"):
            local_starts = [start - self.buffer_latent_start for start in starts]
            pad_short = bool(self.model.tail_hold_latents) and not self.final_segment
            assert self.buffer is not None
        video = self.model._decode_windows(
            self.buffer, local_starts, self.seed, pad_short=pad_short, frame_offset=self.buffer_latent_start
        )
        with self.model._decode_guard("stream window output"):
            if video is not None and self.to_cpu:
                video = video.cpu()
        return video
