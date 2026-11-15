# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas and nv-tlabs/PixelUMM (Apache-2.0 code).
# Source: Hy-Nomas@911cea6458; Copyright 2026 Tencent.

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.models.hy_nomas.attention import RangeAttention

TUBELET = (4, 16, 16)
TUBE_DIM = TUBELET[0] * TUBELET[1] * TUBELET[2] * 3
LATENT_CHANNELS = 32
# A DCAE latent cell is (4, 32, 32) pixels: 1 tubelet in time, 2 x 2 in space.
LATENT_TOKEN_GRID = 2
# Target size of one FP32 Q/K input chunk; intermediate tensors add overhead.
_QK_CHUNK_INPUT_BYTES = 16 * 2**20


@dataclass(frozen=True)
class GenExpertArch:
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

    def __post_init__(self):
        if (
            self.head_dim <= 0
            or self.head_dim % 8
            or self.num_heads <= 0
            or self.num_kv_heads <= 0
            or self.num_heads % self.num_kv_heads
        ):
            raise ValueError(
                "head_dim must be a positive multiple of eight and query heads divisible by positive KV heads"
            )

    @property
    def rope_sections(self) -> tuple[int, int, int]:
        """Head-dim slices rotated by the T, H and W position respectively."""
        return self.head_dim // 2, self.head_dim // 4, self.head_dim // 4


QWEN3_8B_GEN = GenExpertArch()


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def mrope_tables(arch: GenExpertArch, grid: tuple[int, int, int], device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [1, T*H*W, 1, head_dim] for a T/H/W-ordered grid, laid out [T | H | W]."""
    positions = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
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


def apply_mrope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sections) -> torch.Tensor:
    """x [B,N,H,D]; each section is rotated within itself (rotate_half per axis)."""
    rotated = torch.cat([_rotate_half(part) for part in x.split(tuple(sections), dim=-1)], dim=-1)
    return x * cos + rotated * sin


_LOCAL_LAYOUTS: OrderedDict = OrderedDict()


def local_attention_layout(grid: tuple[int, int, int], tile: int, device):
    """``(perm, inverse, block_mask)`` of the sliding-tile attention, or None when the plane has
    at most 3 x 3 tiles (full attention). ``tokens[:, perm]`` is the tile-major order the mask is
    built in; cached per shape."""
    from torch.nn.attention.flex_attention import create_block_mask

    frames, height, width = grid
    tiles_h, tiles_w = -(-height // tile), -(-width // tile)
    if tiles_h <= 3 and tiles_w <= 3:
        return None
    key = (grid, tile, str(device))
    if key not in _LOCAL_LAYOUTS:
        t, h, w = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
        tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
        order = (((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile + (h % tile).flatten()) * tile
        perm = torch.argsort(order + (w % tile).flatten())
        tile_h, tile_w = tile_h[perm], tile_w[perm]
        center_h = tile_h.clamp(1, max(tiles_h - 2, 1))
        center_w = tile_w.clamp(1, max(tiles_w - 2, 1))

        def mask_mod(batch, head, q_idx, kv_idx):
            return ((center_h[q_idx] - tile_h[kv_idx]).abs() <= 1) & ((center_w[q_idx] - tile_w[kv_idx]).abs() <= 1)

        count = frames * height * width
        block_mask = create_block_mask(
            mask_mod,
            None,
            None,
            count,
            count,
            device=device,
            _compile=torch.device(device).type == "cuda",
        )
        _LOCAL_LAYOUTS[key] = (perm, torch.argsort(perm), block_mask)
        while len(_LOCAL_LAYOUTS) > 32:
            _LOCAL_LAYOUTS.popitem(last=False)
    _LOCAL_LAYOUTS.move_to_end(key)
    return _LOCAL_LAYOUTS[key]


class RMSNorm(nn.Module):
    """Qwen3RMSNorm: normalize in fp32, cast back to the input dtype, then scale."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class GenAttention(nn.Module):
    """GQA, split T / HW QK-norm, T/H/W MRoPE; full or sliding-tile attention."""

    def __init__(self, arch: GenExpertArch):
        super().__init__()
        self.arch = arch
        hidden, heads, kv, dim = arch.hidden_size, arch.num_heads, arch.num_kv_heads, arch.head_dim
        self.q_proj = nn.Linear(hidden, heads * dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv * dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv * dim, bias=False)
        self.o_proj = nn.Linear(heads * dim, hidden, bias=False)
        self.q_norm_t = RMSNorm(dim // 2, arch.rms_norm_eps)
        self.q_norm_hw = RMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_t = RMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_hw = RMSNorm(dim // 2, arch.rms_norm_eps)
        self.local_attn = RangeAttention(dim**-0.5)
        self.full_attn = SDPAImpl(
            num_heads=heads,
            num_kv_heads=heads,
            head_size=dim,
            causal=False,
            softmax_scale=dim**-0.5,
        )

    @staticmethod
    def _split_norm(x: torch.Tensor, norm_t: RMSNorm, norm_hw: RMSNorm) -> torch.Tensor:
        t_part, hw_part = x.chunk(2, dim=-1)
        return torch.cat((norm_t(t_part), norm_hw(hw_part)), dim=-1)

    def _prepare_qk(self, x, cos, sin, norm_t, norm_hw, dtype):
        # Bound FP32 QK/RoPE intermediates without splitting the norm reduction
        # dimension. All operations and casts retain their original order.
        batch, tokens, heads, head_dim = x.shape
        fp32_bytes = torch.finfo(torch.float32).bits // 8
        bytes_per_token = batch * heads * head_dim * fp32_bytes
        chunk = max(1, _QK_CHUNK_INPUT_BYTES // bytes_per_token)
        if torch.is_grad_enabled() or tokens <= chunk:
            normalized = self._split_norm(x.float(), norm_t, norm_hw)
            return apply_mrope(normalized, cos, sin, self.arch.rope_sections).to(dtype)
        result = torch.empty_like(x, dtype=dtype)
        for first in range(0, tokens, chunk):
            last = min(first + chunk, tokens)
            normalized = self._split_norm(x[:, first:last].float(), norm_t, norm_hw)
            part = apply_mrope(normalized, cos[:, first:last], sin[:, first:last], self.arch.rope_sections)
            result[:, first:last].copy_(part.to(dtype))
            del normalized, part
        return result

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None) -> torch.Tensor:
        arch = self.arch
        batch, count, _ = x.shape
        q = self.q_proj(x).view(batch, count, arch.num_heads, arch.head_dim)
        k = self.k_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        v = self.v_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        # PixelUMM's generation path keeps q/k in fp32 through QK-norm and RoPE.
        q = self._prepare_qk(q, cos, sin, self.q_norm_t, self.q_norm_hw, v.dtype)
        k = self._prepare_qk(k, cos, sin, self.k_norm_t, self.k_norm_hw, v.dtype)
        if block_mask is not None:
            out = self.local_attn(q, k, v, block_mask=block_mask)
        else:
            repeat = arch.num_heads // arch.num_kv_heads
            k, v = k.repeat_interleave(repeat, dim=2), v.repeat_interleave(repeat, dim=2)
            out = self.full_attn.forward(q, k, v)
        return self.o_proj(out.reshape(batch, count, arch.num_heads * arch.head_dim))


class GenMLP(nn.Module):
    def __init__(self, arch: GenExpertArch):
        super().__init__()
        self.gate_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
        self.up_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
        self.down_proj = nn.Linear(arch.intermediate_size, arch.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GenLayer(nn.Module):
    def __init__(self, arch: GenExpertArch):
        super().__init__()
        self.input_norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.attn = GenAttention(arch)
        self.post_attn_norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.mlp = GenMLP(arch)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin, block_mask)
        return x + self.mlp(self.post_attn_norm(x))


class PixelHead(nn.Module):
    """PixelUMM ``VideoUnpatchifyHead``: RMSNorm(eps=1e-6) -> Linear(hidden -> 3072)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.linear = nn.Linear(hidden, TUBE_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class DCAEVPixelUMMDecoder(nn.Module):
    """``forward(latent [B,32,L,h,w], xt [B,3,4L,32h,32w]) -> x0 prediction``."""

    def __init__(self, arch: GenExpertArch = QWEN3_8B_GEN, *, local_attention_tile: int = 0):
        super().__init__()
        if local_attention_tile < 0:
            raise ValueError(f"local_attention_tile must be >= 0 tokens, got {local_attention_tile}")
        self.arch = arch
        self.local_attention_tile = int(local_attention_tile)
        hidden = arch.hidden_size
        self.embed = nn.Linear(TUBE_DIM, hidden)
        self.cond_proj = nn.Linear(LATENT_CHANNELS, LATENT_TOKEN_GRID**2 * hidden)
        self.layers = nn.ModuleList(GenLayer(arch) for _ in range(arch.num_layers))
        self.final_norm = RMSNorm(hidden, arch.rms_norm_eps)
        self.head = PixelHead(hidden)

    @staticmethod
    def pixel_shape(latent: torch.Tensor) -> tuple[int, int, int, int, int]:
        batch, _, frames, height, width = latent.shape
        return batch, 3, frames * TUBELET[0], height * 32, width * 32

    def _condition_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = latent.shape
        g = LATENT_TOKEN_GRID
        x = self.cond_proj(latent.permute(0, 2, 3, 4, 1))  # [B,L,h,w,4*hidden]
        x = x.view(batch, frames, height, width, g, g, self.arch.hidden_size)
        return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, -1, self.arch.hidden_size)

    def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"expected DCAE latent [B,{LATENT_CHANNELS},L,h,w], got {tuple(latent.shape)}")
        if tuple(xt.shape) != self.pixel_shape(latent):
            raise ValueError(f"xt {tuple(xt.shape)} != expected {self.pixel_shape(latent)}")
        dtype = self.embed.weight.dtype
        context = (
            torch.autocast(device_type=latent.device.type, dtype=dtype)
            if dtype in (torch.bfloat16, torch.float16)
            else nullcontext()
        )
        _, _, frames, height, width = latent.shape
        grid = (frames, height * LATENT_TOKEN_GRID, width * LATENT_TOKEN_GRID)
        with context:
            tokens = self.embed(patchify(xt.to(dtype))) + self._condition_tokens(latent.to(dtype))
            cos, sin = mrope_tables(self.arch, grid, latent.device)
            layout = (
                local_attention_layout(grid, self.local_attention_tile, latent.device)
                if self.local_attention_tile
                else None
            )
            block_mask = None
            if layout is not None:
                perm, inverse, block_mask = layout
                tokens, cos, sin = tokens[:, perm], cos[:, perm], sin[:, perm]
            for layer in self.layers:
                tokens = layer(tokens, cos, sin, block_mask)
            tokens = self.final_norm(tokens)
            if layout is not None:
                tokens = tokens[:, inverse]
        # The source runs the head after leaving autocast (forward_features returns first).
        return unpatchify(self.head(tokens), grid)
