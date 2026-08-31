# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# PixelDiT T2I — consolidated network architecture.
# Verbatim copy from the original PixelDiT repo, merged into a single file.
# Sources:
#   pixdit_core/modules.py        — building blocks (RMSNorm, RoPE, attention, etc.)
#   pixdit_core/pixeldit_c2i.py   — PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock
#   pixdit_core/pixeldit_t2i.py   — MMDiT joint attention, encoder-decoder, PixDiT_T2I
#
# Only import statements were changed (everything is now local). Logic is unchanged.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

# =============================================================================
# From pixdit_core/modules.py
# =============================================================================


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros(extra_tokens, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a tensor of positions to be encoded: size (M,)
    out: (M, D)

    Pure-torch port of the original numpy implementation (float64 intermediate for
    bit-for-bit parity with the released checkpoints). Stays traceable by torch.compile.
    """
    assert embed_dim % 2 == 0
    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos)
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1).to(torch.float64)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def _interleave_cos_sin(x_freqs: torch.Tensor, y_freqs: torch.Tensor) -> torch.Tensor:
    """Pack per-axis rotation angles into a real (cos, sin) RoPE tensor.

    Returns `[N, (dim//4)*2, 2]` where `[..., 0]`/`[..., 1]` are cos/sin and the
    x/y axes are interleaved (element 2j = x-axis, 2j+1 = y-axis), matching the
    layout `apply_rotary_emb` pairs with the (real, imag) halves of q/k. This is
    the real-valued equivalent of the old `torch.polar` complex representation —
    same math, but traceable by torch.compile (Dynamo can't handle complex ops).
    """
    angles = torch.stack([x_freqs, y_freqs], dim=-1).reshape(x_freqs.shape[0], -1)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    return _interleave_cos_sin(x_freqs, y_freqs)


def precompute_freqs_cis_2d_ntk(
    dim: int,
    height: int,
    width: int,
    ref_grid_h: int,
    ref_grid_w: int,
    theta: float = 10000.0,
    scale: float = 16.0,
):
    """NTK-aware 2D RoPE.  Identical to precompute_freqs_cis_2d when
    height == ref_grid_h and width == ref_grid_w.  For other resolutions
    the base theta is scaled per-axis following the NTK-aware formula:
        ntk_factor = (current / ref) ** (dim_axis / (dim_axis - 2))
        theta_axis = theta * ntk_factor
    where dim_axis = dim // 2 (half the head dim per spatial axis).
    """
    dim_axis = dim // 2  # each axis gets dim//4 complex pairs → dim//2 real dims
    h_scale = height / ref_grid_h
    w_scale = width / ref_grid_w
    h_ntk = h_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    w_ntk = w_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    h_theta = theta * h_ntk
    w_theta = theta * w_ntk

    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)

    freqs_w = 1.0 / (w_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_h = 1.0 / (h_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    x_freqs = torch.outer(x_pos, freqs_w).float()
    y_freqs = torch.outer(y_pos, freqs_h).float()
    return _interleave_cos_sin(x_freqs, y_freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # freqs_cis: real (cos, sin) tensor [N, head_dim//2, 2] (see `_interleave_cos_sin`).
    # Apply RoPE as an explicit 2x2 rotation on each (real, imag) pair of q/k —
    # bit-equivalent to the old complex multiply, but traceable by torch.compile.
    cos = freqs_cis[None, :, None, :, 0]  # [1, N, 1, head_dim//2]
    sin = freqs_cis[None, :, None, :, 1]

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_r, x_i = x_[..., 0], x_[..., 1]
        out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1).flatten(-2)
        return out.type_as(x)

    return _rotate(xq), _rotate(xk)


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # vLLM-Omni Attention layer: dispatches Ulysses / Ring / AllGather-KV
        # automatically when sequence parallelism is active. When SP is off it
        # is unused (the original SDPA path below keeps identical numerics).
        self.vattn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.scale,
            prefix="pid.rotary_attn",
        )

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        if self._sp_active():
            # SP: sequence already sharded along N by _sp_plan; vLLM-Omni
            # Attention handles Ulysses (all-to-all) / Ring / AllGather-KV.
            if mask is not None:
                logger.warning_once("RotaryAttention: attention mask ignored under sequence parallelism.")
            out = self.vattn(q, k, v, None)
            out = out.reshape(B, N, C)
        else:
            q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
            v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

            out = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

            out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _sp_active(self) -> bool:
        return is_forward_context_available() and bool(get_forward_context().sp_active)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


# =============================================================================
# From pixdit_core/pixeldit_c2i.py (PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock)
# =============================================================================


class PatchTokenEmbedder(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache = dict()

    def _fetch_pixel_pos_patch(self, patch_size: int, device, dtype):
        key = ("patch", patch_size)
        if key in self._pos_cache:
            pe = self._pos_cache[key]
            return pe.to(device=device, dtype=dtype)
        pos = get_2d_sincos_pos_embed(self.hidden_size_output, patch_size).to(device=device, dtype=dtype)  # [P2, D]
        self._pos_cache[key] = pos
        return pos

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        key = ("image", height, width)
        if key in self._pos_cache:
            pe = self._pos_cache[key]
            return pe.to(device=device, dtype=dtype)
        if height == width:
            pos = get_2d_sincos_pos_embed(self.hidden_size_output, height).to(device=device, dtype=dtype)  # [H*W, D]
        else:
            # Build a non-square grid (H x W) and compute 2D sin/cos embedding.
            grid_h = torch.arange(height, dtype=torch.float32)
            grid_w = torch.arange(width, dtype=torch.float32)
            grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # w first to match existing convention
            grid = torch.stack(grid, dim=0).reshape(2, 1, height, width)
            pos = get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid).to(device=device, dtype=dtype)
        self._pos_cache[key] = pos
        return pos

    def forward(
        self,
        inputs: torch.Tensor,
        img_height: int = None,
        img_width: int = None,
        patch_size: int = None,
    ):
        # Two modes:
        # 1) Legacy patch : inputs [B*L, P2, C] -> add 2D sincos within patch (P2 = patch_size^2)
        # 2) Image : inputs [B, C, H, W] -> patchify inside and add full-image pixel-space sincos sampled per patch
        if inputs.dim() == 3:
            # Legacy: [B*L, P2, C]
            batch_tokens, p2, _ = inputs.shape
            patch_sz = int(p2**0.5)
            pos = self._fetch_pixel_pos_patch(patch_sz, inputs.device, inputs.dtype)  # [P2, D]
            x = self.proj(inputs)
            x = x + pos.unsqueeze(0)
            return x
        elif inputs.dim() == 4:
            # Image mode: [B, C, H, W]
            assert img_height is not None and img_width is not None and patch_size is not None, (
                "Need H, W, patch_size for image mode"
            )
            B, C, H, W = inputs.shape
            assert H == img_height and W == img_width, "Input spatial size mismatch"
            assert (H % patch_size == 0) and (W % patch_size == 0), "H and W must be divisible by patch_size"
            Hs, Ws = H // patch_size, W // patch_size
            P2 = patch_size * patch_size
            # linear proj per pixel
            x = inputs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            x = self.proj(x)  # [B, H, W, D]
            # full-image pixel-space pos
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)  # [H*W, D]
            pos_full = pos_full.view(H, W, self.hidden_size_output)
            # add pos at image grid then patchify to [B*L, P2, D]
            x = x + pos_full.unsqueeze(0)
            x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, Hs, Ws, ps, ps, D]
            x = x.view(B * Hs * Ws, P2, self.hidden_size_output)
            return x
        else:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B*L,P2,C] or [B,C,H,W]")


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: int | None = None,
        attn_num_heads: int | None = None,
        rope_mode: str = "original",
        rope_ref_grid_h: int = 32,
        rope_ref_grid_w: int = 32,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        assert self.attn_dim % self.num_heads == 0, "pixel attention hidden size must be divisible by pixel num_heads"
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, 6 * self.pixel_dim * p2, bias=True))
        self._pos_cache = dict()

    def _fetch_pos(self, height: int, width: int, device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        head_dim = self.attn_dim // self.num_heads
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(
                device
            )
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        image_height: int,
        image_width: int,
        patch_size: int,
        mask=None,
        patch_count: int | None = None,
    ) -> torch.Tensor:
        # x: [B*L, P2, C] where L = Hs*Ws patch tokens. Under sequence
        # parallelism `patch_count` carries the *local* (sharded) patch count.
        BL, P2, C = x.shape
        if C != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {C}")
        assert patch_size == self.patch_size, "PiTBlock expects fixed patch_size"
        assert P2 == patch_size * patch_size, "Token count per patch must equal patch_size^2"
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0), (
            "H and W must be divisible by patch_size"
        )
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = patch_count if patch_count is not None else Hs * Ws
        assert s_cond.shape[0] == BL, "s_cond batch must match x batch"
        assert BL % L == 0, "Total sequences must be a multiple of patch count"
        B = BL // L
        # adaLN per pixel (within patch): params
        cond_params = self.adaLN_modulation(s_cond)  # [BL, 6*pixel_dim*P2]
        cond_params = cond_params.view(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, L, self.attn_dim)
        # attention across patch tokens (L)
        pos_comp = self._fetch_pos(Hs, Ws, x.device)
        attn_out = self.attn(x_comp, pos_comp, mask)  # [B, L, attn_dim]
        attn_flat = self.expand_from_attn(attn_out.view(B * L, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        # residual & MLP locally
        x = x + gate_msa * attn_exp
        mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out
        return x


# =============================================================================
# From pixdit_core/pixeldit_t2i.py
# =============================================================================


class PiDPrepare(nn.Module):
    """Sequence-parallel shard boundary for :class:`PixDiT_T2I`.

    Produces the three tensors that must be sharded *together* along the
    patch-token sequence ``L`` so that tokens and their RoPE freqs stay
    aligned:

      - ``x_patches`` ``[B, L, patch_size**2 * C]`` (patch-token condition path)
      - ``x_pixels``  ``[B*L, P2, pixel_hidden]``   (pixel pathway)
      - ``pos_img``   ``[L, ...]``                  (image-stream RoPE freqs)

    Pure-function wrapper (holds no submodules) to avoid a cyclic parent
    reference back to :class:`PixDiT_T2I`.
    """

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        hs: int,
        ws: int,
        patch_size: int,
        pixel_embedder: nn.Module,
        fetch_pos,
    ):
        x_patches = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
        x_pixels = pixel_embedder(x, img_height=h, img_width=w, patch_size=patch_size)
        pos_img = fetch_pos(hs, ws, x.device)
        return x_patches, x_pixels, pos_img


class MMDiTJointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Separate QKV projections for image (x) and text (y) streams
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Per-stream QK normalization (head-wise)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        # Output projections for each stream
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)

        self.vattn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            prefix="pid.mmdit_attn",
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, Nx, C] image stream (Nx sharded under SP)
        y: torch.Tensor,  # [B, Ny, C] text stream (always replicated)
        pos_img: torch.Tensor,  # [Nx_local, head_dim/2] RoPE freqs (sharded with x)
        pos_txt: torch.Tensor = None,  # [Ny, head_dim/2] RoPE freqs for text (optional)
        attn_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape
        assert B == By and C == Cy, "x and y must share batch and channel dims"

        # QKV for image
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]  # [B, Nx, H, Hc]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        # QKV for text
        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]  # [B, Ny, H, Hc]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        # Image RoPE
        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        if self._sp_active():
            # SP: image stream sharded along Nx, text stream replicated as a
            # joint-front KV context. vLLM-Omni Attention internally handles
            # Ulysses (all-to-all) / Ring / AllGather-KV for the image query.
            # The text stream's own attention output is not produced (same as
            # Qwen-Image under SP) -> its residual stays unchanged.
            if attn_mask is not None:
                logger.warning_once("MMDiTJointAttention: attention mask ignored under sequence parallelism.")
            md = AttentionMetadata(
                joint_query=qy,
                joint_key=ky,
                joint_value=vy,
                joint_strategy="front",
            )
            # The Ulysses/Ring strategies re-concatenate the joint text to the
            # front of the output ([Ny + Nx, H, Hc]), so drop the text part and
            # keep only the (sharded) image tokens.
            out_x = self.vattn(qx, kx, vx, md)  # [B, Ny + Nx, H, Hc]
            out_x = out_x[:, Ny:, :, :]
            out_x = out_x.reshape(B, Nx, C)
            out_y = torch.zeros_like(y)
        else:
            # SDPA expects [B, H, S, Hc]; build joint sequence [text, image].
            qx = qx.transpose(1, 2)
            kx = kx.transpose(1, 2)
            vx = vx.transpose(1, 2)

            qy = qy.transpose(1, 2)  # [B, H, Ny, Hc]
            ky = ky.transpose(1, 2)
            vy = vy.transpose(1, 2)

            q_joint = torch.cat([qy, qx], dim=2)  # [B, H, Ny + Nx, Hc]
            k_joint = torch.cat([ky, kx], dim=2)  # [B, H, Ny + Nx, Hc]
            v_joint = torch.cat([vy, vx], dim=2)

            out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
            # Split back to [text, image]
            out_y = out_joint[:, :, :Ny, :]
            out_x = out_joint[:, :, Ny:, :]

            # Merge heads
            out_y = out_y.transpose(1, 2).reshape(B, Ny, C)
            out_x = out_x.transpose(1, 2).reshape(B, Nx, C)

        # Output projections
        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y

    def _sp_active(self) -> bool:
        return is_forward_context_available() and bool(get_forward_context().sp_active)


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, ada_ln_modulation_img=None, ada_ln_modulation_txt=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        # Per-stream norms
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)

        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)

        # Per-stream AdaLN modulation
        self.adaLN_modulation_img = (
            ada_ln_modulation_img
            if ada_ln_modulation_img is not None
            else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )
        self.adaLN_modulation_txt = (
            ada_ln_modulation_txt
            if ada_ln_modulation_txt is not None
            else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None):
        # c: [B, 1, C] typically, broadcast across tokens
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(
            6, dim=-1
        )
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(
            6, dim=-1
        )

        # 1) Joint attention with dual-stream
        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        # 2) Per-stream MLP with AdaLN
        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


# Main T2I network: PixDiT_T2I
# =============================================================================


class PixDiT_T2I(nn.Module):
    # Sequence parallelism plan (vLLM-Omni). Patch/pixel token streams and
    # their image-stream RoPE are sharded together along the patch sequence L;
    # the pixel-block RoPE is sharded at the attention input; the output is
    # gathered at final_layer (before fold).
    _sp_plan = {
        "pid_prepare": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: SequenceParallelInput(split_dim=0, expected_dims=3, split_output=True),
            2: SequenceParallelInput(split_dim=0, expected_dims=3, split_output=True),
        },
        "pixel_blocks.*.attn": {
            "pos": SequenceParallelInput(split_dim=0, expected_dims=3),
        },
        "final_layer": SequenceParallelOutput(gather_dim=0, expected_dims=3),
    }

    def __init__(
        self,
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        # NTK-aware RoPE: set rope_mode="ntk_aware" and provide the reference
        # pixel resolution used during training.  When the actual grid size
        # differs from ref, the base theta is scaled per-axis.
        rope_mode: str = "original",  # "original" | "ntk_aware"
        rope_ref_h: int = 1024,
        rope_ref_w: int = 1024,
        repa_encoder_index: int = -1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_h // self.patch_size
        self.rope_ref_grid_w = rope_ref_w // self.patch_size
        self.repa_encoder_index = int(repa_encoder_index)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0 to retain the pixel pathway")

        # Embedders
        self.pixel_embedder = PixelTokenEmbedder(in_channels, self.pixel_hidden_size)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden_size))

        # Blocks
        # Shared AdaLN modulator for conditional blocks (optional)
        self._shared_cond_adaln = None
        self._shared_cond_adaln_img = None
        self._shared_cond_adaln_txt = None
        self.patch_blocks = nn.ModuleList(
            [
                MMDiTBlockT2I(
                    self.hidden_size,
                    self.num_groups,
                    ada_ln_modulation_img=self._shared_cond_adaln_img,
                    ada_ln_modulation_txt=self._shared_cond_adaln_txt,
                )
                for _ in range(self.patch_depth)
            ]
        )
        # Remove AdaLN-based text refinement; PixDiT keeps cross-attn-only text handling
        self.text_refine_blocks = None
        self.pixel_attn_hidden_size = (
            int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        )
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                    rope_mode=self.rope_mode,
                    rope_ref_grid_h=self.rope_ref_grid_h,
                    rope_ref_grid_w=self.rope_ref_grid_w,
                )
                for _ in range(self.pixel_depth)
            ]
        )

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        # Sequence-parallel shard boundary (no params; a pure wrapper that
        # produces the patch/pixel/RoPE tensors the _sp_plan shards together).
        self.pid_prepare = PiDPrepare()

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()  # cache for 1D text RoPE
        self.last_repa_tokens = None

    def _sp_active(self) -> bool:
        return is_forward_context_available() and bool(get_forward_context().sp_active)

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        head_dim = self.hidden_size // self.num_groups
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(
                device
            )
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self.precompute_pos[(height, width)] = pos
        return pos

    def fetch_pos_text(self, length, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        # Build 1D RoPE freqs for text stream using the same per-head dim as image.
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)  # [length,1]
        angles = positions * freqs.unsqueeze(0)  # [length, head_dim//2]
        # Real (cos, sin) layout [length, head_dim//2, 2] consumed by `apply_rotary_emb`.
        freqs_cis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    @torch.no_grad()
    def precompute_positional_caches(self, image_height, image_width, text_length, device=None, pixel_dtype=None):
        """Eagerly warm every positional cache for a fixed (H, W, text_length).

        Called once before `torch.compile` so the compiled `forward` only ever hits
        the cache-return branch — no RoPE/sincos recompute and no dict mutation inside
        the traced graph (both would otherwise force graph breaks). Standard SR path
        only (no encoder-decoder / token-shuffle).
        """
        device = device or next(self.parameters()).device
        if pixel_dtype is None:
            pixel_dtype = next(self.parameters()).dtype
        Hs, Ws = image_height // self.patch_size, image_width // self.patch_size
        self.fetch_pos(Hs, Ws, device)
        if self.use_text_rope:
            self.fetch_pos_text(text_length, device)
        self.pixel_embedder._fetch_pixel_pos_image(image_height, image_width, device, pixel_dtype)
        for blk in self.pixel_blocks:
            blk._fetch_pos(Hs, Ws, device)

    def forward(self, x, t, y, s=None, mask=None):
        B, _, H, W = x.shape
        # Derive grid token count deterministically from spatial size
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        # Sequence-parallel shard boundary: the post-forward hook splits the
        # patch tokens, pixel tokens and image-stream RoPE together along L so
        # tokens and their freqs stay aligned. Without SP this returns the full
        # tensors (L_local == L) and the flow is unchanged.
        x_patches, x_pixels, pos = self.pid_prepare(
            x, H, W, Hs, Ws, self.patch_size, self.pixel_embedder, self.fetch_pos
        )
        L_local = x_patches.shape[1]

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        # Text tokens -> project to hidden_size and add learned pos
        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        # PixDiT design: no AdaLN modulation applied on text stream
        condition = torch.nn.functional.silu(t_emb)

        # Condition blocks on patch tokens with MM-DiT joint attention to text tokens
        pad = None
        pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
        if mask is not None and isinstance(mask, torch.Tensor):
            m = mask
            while m.dim() > 2 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 3 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 2:
                pad = m == 0

        if s is None:
            s0 = self.s_embedder(x_patches)
            self.last_repa_tokens = None
            s_main = s0
            attn_mask_joint = None
            if pad is not None:
                L_img_curr = s_main.shape[1]
                pad_img = torch.zeros((B, L_img_curr), dtype=torch.bool, device=x.device)
                pad_txt = (
                    pad[:, :Ltxt]
                    if pad.size(1) >= Ltxt
                    else torch.nn.functional.pad(pad, (0, Ltxt - pad.size(1)), value=True)
                )
                attn_mask_joint = torch.cat([pad_txt, pad_img], dim=1).view(B, 1, 1, Ltxt + L_img_curr)

            for i in range(self.patch_depth):
                s_main, y_emb = self.patch_blocks[i](s_main, y_emb, condition, pos, pos_txt, attn_mask_joint)
                if 0 < self.repa_encoder_index == (i + 1):
                    self.last_repa_tokens = s_main
            s = torch.nn.functional.silu(t_emb + s_main)
        # If no valid tap index is specified, expose last conditional output
        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        # Ensure the patch token length matches the (sharded) local grid
        batch_size, length, _ = s.shape
        if length != L_local:
            if length > L_local:
                s = s[:, :L_local, :]
            else:
                pad_len = L_local - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)
            length = L_local

        # Pixel pathway (x_pixels already produced by pid_prepare; under SP it
        # is sharded along B*L, so the local patch count is passed to PiTBlock)
        s_cond = s.view(B * L_local, self.hidden_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask, patch_count=L_local)

        # Project back to image and fold. The final_layer gather hook restores
        # the full patch count (L) before the fold.
        x_pixels = self.final_layer(x_pixels)  # [B*L, P2, C]
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
