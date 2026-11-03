# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent LiDAR encoder and decoder loaded from the unified VAE.

The checkpoint root contains ``lidar_vae/config.json`` and safetensors
weights (optionally sharded). Both components load their own weights through
``DiffusersPipelineLoader`` and run in FP32. No multiview
transformer or generation pipeline is required::

    from vllm_omni.diffusion.models.cosmos3.lidar_vae import Cosmos3LidarDecoder, Cosmos3LidarEncoder

    root = Path("/models/cosmos3")
    config = json.loads((root / "lidar_vae/config.json").read_text())
    device = torch.device("cuda")
    encoder = Cosmos3LidarEncoder.from_pretrained(str(root), config, device)
    decoder = Cosmos3LidarDecoder.from_pretrained(str(root), config, device)
    latents = encoder(frames)  # FP32 physical frames: [B, 3, T, 128, 1800]
    reconstructed = decoder(latents)  # [B, 3, T, 128, 1800]

The latent shape is ``[B, latent_channels, T, 8, 113]``; an unbatched
``[3, T, 128, 1800]`` input encodes with ``B = 1``. Decoder channels are
range in metres, unit intensity, and validity. Full-size neighborhood
attention requires CUDA; CPU execution is for small correctness fixtures.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Any, Literal

import einops
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn.modules.utils import _pair
from typing_extensions import Self

from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch
from vllm_omni.diffusion.models.cosmos3.norm import RMSNorm

if TYPE_CHECKING:
    from vllm.config.load import LoadConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader


# =============================================================================
# Polar coordinates
# =============================================================================


def generate_polar_coords(
    height: int, width: int, device: torch.device | str = "cpu"
) -> torch.Tensor:  # returns [1,2,H,W]
    """Build spherical angles for a range map.

    phi: elevation in (-pi/2, pi/2], decreasing with row; row 0 is +pi/2.
    theta: azimuth in (-pi, pi], decreasing with column; column 0 is +pi.
    Angles span the model width linearly (padded columns are not wrapped),
    matching the reference buffer stored in the checkpoint.
    """
    phi = (0.5 - torch.arange(height, device=device) / height) * torch.pi  # [H]
    theta = (1 - torch.arange(width, device=device) / width) * 2 * torch.pi - torch.pi  # [W]
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")  # [H,W], [H,W]
    angles = torch.stack([phi, theta])  # [2,H,W]
    return angles[None]  # [1,2,H,W]


# =============================================================================
# 3D rotary positional embedding
# =============================================================================


class VideoRopePosition3DEmb(nn.Module):
    """3D RoPE producing angles for a ``(T, H, W)`` token grid.

    The head dimension is split into three chunks ``(dim_t, dim_h, dim_w)``.
    Each axis contributes its own rotary frequency band, and the resulting angles are duplicated
    across the two rotation halves (GPT-NeoX style) so that
    :func:`apply_rotary_emb` can rotate Q/K with a single tensor.

    For a query at position ``p_q = (t_q, h_q, w_q)`` and a key at
    ``p_k = (t_k, h_k, w_k)``, the dot product after rotation depends only on
    ``p_q - p_k``. If the query and key share the same value along some axis
    (e.g. a factored *spatial* attention block where both are on the same
    frame), the rotation along that axis cancels out -- so this single 3D
    RoPE specializes correctly to 2D RoPE for spatial-only blocks and 1D RoPE
    for temporal-only blocks.

    Args:
        head_dim: Per-head channel count. Must satisfy
            ``head_dim - 2 * (head_dim // 6 * 2) > 0``.
        len_t: Maximum supported temporal grid size.
        len_h: Maximum supported grid height.
        len_w: Maximum supported grid width. Frequencies are cached for these
            maxima; ``forward`` accepts any smaller ``(T, H, W)``.
        base_theta: RoPE base (``10000.0`` matches LLaMA/Cosmos).
    """

    def __init__(
        self,
        *,
        head_dim: int,
        len_t: int,
        len_h: int,
        len_w: int,
        base_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        dim = head_dim
        dim_h = (dim // 6) * 2
        dim_w = dim_h
        dim_t = dim - dim_h - dim_w
        assert dim_t >= 2, f"head_dim={dim} too small for 3D RoPE (dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w})"
        self.head_dim = dim
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.max_t = len_t
        self.max_h = len_h
        self.max_w = len_w

        idx_h = torch.arange(0, dim_h, 2, dtype=torch.float32) / dim_h
        idx_w = torch.arange(0, dim_w, 2, dtype=torch.float32) / dim_w
        idx_t = torch.arange(0, dim_t, 2, dtype=torch.float32) / dim_t
        self.register_buffer("freqs_h", 1.0 / (base_theta**idx_h), persistent=False)
        self.register_buffer("freqs_w", 1.0 / (base_theta**idx_w), persistent=False)
        self.register_buffer("freqs_t", 1.0 / (base_theta**idx_t), persistent=False)
        self.register_buffer("seq_h", torch.arange(len_h, dtype=torch.float32), persistent=False)
        self.register_buffer("seq_w", torch.arange(len_w, dtype=torch.float32), persistent=False)
        self.register_buffer("seq_t", torch.arange(len_t, dtype=torch.float32), persistent=False)

    def forward(self, num_frames: int, height: int, width: int) -> torch.Tensor:  # returns [T*H*W,D]
        """Returns rope angles of shape ``(T*H*W, head_dim)``.

        Each half of the last dim duplicates the per-axis angles
        ``[t_half | h_half | w_half | t_half | h_half | w_half]`` so that
        :func:`apply_rotary_emb` (GPT-NeoX style) can rotate Q/K via
        ``[x_lo, x_hi] @ rot``.
        """
        assert num_frames <= self.max_t and height <= self.max_h and width <= self.max_w, (
            f"Requested ({num_frames},{height},{width}) exceeds cached ({self.max_t},{self.max_h},{self.max_w})"
        )
        ang_t = torch.outer(self.seq_t[:num_frames], self.freqs_t)  # [T,Dt/2]
        ang_h = torch.outer(self.seq_h[:height], self.freqs_h)  # [H,Dh/2]
        ang_w = torch.outer(self.seq_w[:width], self.freqs_w)  # [W,Dw/2]
        ang_t_b = ang_t.view(num_frames, 1, 1, -1).expand(num_frames, height, width, -1)  # [T,H,W,Dt/2]
        ang_h_b = ang_h.view(1, height, 1, -1).expand(num_frames, height, width, -1)  # [T,H,W,Dh/2]
        ang_w_b = ang_w.view(1, 1, width, -1).expand(num_frames, height, width, -1)  # [T,H,W,Dw/2]
        half = torch.cat([ang_t_b, ang_h_b, ang_w_b], dim=-1)  # [T,H,W,D/2]
        full = torch.cat([half, half], dim=-1)  # [T,H,W,D]
        return full.reshape(num_frames * height * width, -1)  # [T*H*W,D]


def apply_rotary_emb(
    x: torch.Tensor, rope_emb: torch.Tensor
) -> torch.Tensor:  # x: [...,S,N,D], rope_emb: [S,D], returns [...,S,N,D]
    """Apply half-split rotation using the shared helper and duplicated 3D angles."""
    angles = rope_emb[:, : x.shape[-1] // 2].to(x.dtype)
    return apply_rotary_emb_torch(x, angles.cos(), angles.sin(), interleaved=False)


# =============================================================================
# Neighborhood attention
# =============================================================================


# Inference-only, inward-shifted 2D neighborhood attention for the LiDAR VAE.
#
# Only geometry and compiled code are shared across requests. Spatial padding is
# owned by the caller; masks describe the actual (possibly padded) input grid.

_BLOCK_SIZE = 64
_QUERY_CHUNK_SIZE = 4096
_CACHE_SIZE = 32
_KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "num_warps": 4,
    "num_stages": 1,
    "USE_TMA": False,
    "FORCE_USE_FLEX_ATTENTION": True,
    # Triton's dot precision is local to this operator, including the P @ V
    # multiplication. Do not change the surrounding model's TF32 policy.
    "FLOAT32_PRECISION": "'ieee'",
}


def _attention_pair(value, name: str) -> tuple[int, int]:
    pair = tuple(value) if isinstance(value, tuple | list) else (value, value)
    if len(pair) != 2 or any(type(v) is not int or v < 1 for v in pair):
        raise ValueError(f"LiDAR attention {name} must contain two positive integers, got {value!r}.")
    return pair


def _axis_start(q, length: int, kernel: int, dilation: int):
    residue = q % dilation
    group_length = (length - 1 - residue) // dilation + 1
    return torch.minimum((q // dilation - kernel // 2).clamp(min=0), group_length - kernel)


def _mask_predicate(height: int, width: int, kernel: tuple[int, int], dilation: tuple[int, int]):
    def mask_mod(batch, head, q, p):
        qh, qw = q // width, q % width
        ph, pw = p // width, p % width
        sh = _axis_start(qh, height, kernel[0], dilation[0])
        sw = _axis_start(qw, width, kernel[1], dilation[1])
        return (
            (q < height * width)
            & (p < height * width)
            & (qh % dilation[0] == ph % dilation[0])
            & (qw % dilation[1] == pw % dilation[1])
            & (ph // dilation[0] >= sh)
            & (ph // dilation[0] < sh + kernel[0])
            & (pw // dilation[1] >= sw)
            & (pw // dilation[1] < sw + kernel[1])
        )

    return mask_mod


@lru_cache(maxsize=_CACHE_SIZE)
def _get_block_mask(
    device: torch.device, height: int, width: int, kernel: tuple[int, int], dilation: tuple[int, int]
) -> BlockMask:
    for length, k, d in zip((height, width), kernel, dilation):
        # Every residue group must contain the entire window, including the
        # shorter groups when the axis is not divisible by dilation.
        if length < k * d:
            raise ValueError(f"LiDAR attention axis length {length} is smaller than kernel {k} * dilation {d}.")
    sequence = height * width
    blocks = math.ceil(sequence / _BLOCK_SIZE)
    counts = torch.empty((blocks,), dtype=torch.int32, device=device)
    indices = torch.empty((blocks, blocks), dtype=torch.int32, device=device)
    dh = torch.arange(kernel[0], device=device)
    dw = torch.arange(kernel[1], device=device)
    for first in range(0, sequence, _QUERY_CHUNK_SIZE):
        q = torch.arange(first, min(first + _QUERY_CHUNK_SIZE, sequence), device=device)
        qh, qw = q // width, q % width
        ph = (_axis_start(qh, height, kernel[0], dilation[0])[:, None] + dh) * dilation[0]
        ph = ph + (qh % dilation[0])[:, None]
        pw = (_axis_start(qw, width, kernel[1], dilation[1])[:, None] + dw) * dilation[1]
        pw = pw + (qw % dilation[1])[:, None]
        key_blocks = ((ph[:, :, None] * width + pw[:, None, :]) // _BLOCK_SIZE).flatten(1)
        row = (q - first) // _BLOCK_SIZE
        rows = math.ceil(q.numel() / _BLOCK_SIZE)
        present = torch.zeros((rows, blocks), dtype=torch.bool, device=device)
        present.view(-1).scatter_(0, (row[:, None] * blocks + key_blocks).flatten(), True)
        start = first // _BLOCK_SIZE
        counts[start : start + rows] = present.sum(-1, dtype=torch.int32)
        # Full-width contiguous int32 metadata matches FlexAttention's native
        # layout. Only the first `counts[row]` entries are visited by the kernel.
        indices[start : start + rows] = present.to(torch.int8).argsort(descending=True, stable=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        counts[None, None],
        indices[None, None],
        BLOCK_SIZE=(_BLOCK_SIZE, _BLOCK_SIZE),
        mask_mod=_mask_predicate(height, width, kernel, dilation),
        seq_lengths=(sequence, sequence),
        compute_q_blocks=False,
    )


def _run_attention(query, key, value, block_mask, scale):
    head_dim = query.shape[-1]
    # Pad small head dimensions to 16 for CUDA FlexAttention. Zero padding
    # preserves Q @ K and P @ V with an explicit scale.
    if head_dim < 16:
        query, key, value = (F.pad(t, (0, 16 - head_dim)) for t in (query, key, value))
    output = flex_attention(query, key, value, block_mask=block_mask, scale=scale, kernel_options=_KERNEL_OPTIONS)
    return output[..., :head_dim] if head_dim < 16 else output


@lru_cache(maxsize=_CACHE_SIZE)
def _get_compiled_runner(device, shape, kernel, dilation, scale):
    # Dynamo caches by code object. A new wrapper around the same code would
    # still hit its recompile limit across decoder levels and partial chunks.
    # Give each static specialization its own code object, without capturing
    # tensors or a mask (which would keep evicted device allocations alive).
    runner = FunctionType(_run_attention.__code__.replace(), globals(), _run_attention.__name__)
    return torch.compile(runner, fullgraph=True, dynamic=False)


def neighborhood_attention_2d(query, key, value, *, kernel_size, dilation, scale):
    """Attend to a fixed 2D window on FP32 [B,H,W,heads,head_dim] tensors.

    CUDA always uses compiled sparse attention and propagates failures. Eager
    CPU FlexAttention is available only for small correctness fixtures, since
    PyTorch's eager implementation materializes token-level scores.
    """
    if query.ndim != 5 or any(t.shape != query.shape for t in (key, value)) or min(query.shape) < 1:
        raise ValueError("LiDAR attention requires matching nonempty [B,H,W,heads,head_dim] tensors.")
    if any(t.dtype != torch.float32 or t.device != query.device for t in (query, key, value)):
        raise ValueError("LiDAR attention requires FP32 query, key, and value on the same device.")
    if query.device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"LiDAR FlexAttention requires CUDA (or CPU correctness fixtures), got {query.device}.")
    kernel, dilation = _attention_pair(kernel_size, "kernel_size"), _attention_pair(dilation, "dilation")
    batch, height, width, heads, dim = query.shape
    if query.device.type == "cpu" and height * width > 4096:
        raise RuntimeError("Eager CPU LiDAR FlexAttention is limited to 4096 tokens; use CUDA for production grids.")
    # Tensor.device resolves an unindexed 'cuda' to the actual device, so an
    # offloaded/reloaded module never reuses another GPU's mask or runner.
    try:
        mask = _get_block_mask(query.device, height, width, kernel, dilation)
        q, k, v = (
            t.permute(0, 3, 1, 2, 4).reshape(batch, heads, height * width, dim).contiguous()
            for t in (query, key, value)
        )
        runner = (
            _get_compiled_runner(query.device, tuple(query.shape), kernel, dilation, scale)
            if query.is_cuda
            else _run_attention
        )
        with torch.autocast(device_type=query.device.type, enabled=False):
            output = runner(q, k, v, mask, scale)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Cosmos3 LiDAR FlexAttention failed on {query.device} for shape {tuple(query.shape)}, "
            f"kernel={kernel}, dilation={dilation}; compiled sparse attention is required on CUDA."
        ) from exc
    return output.transpose(1, 2).reshape(batch, height, width, heads, dim).contiguous()


# =============================================================================
# Transformer VAE blocks
# =============================================================================


# Inference-only LiDAR VAE: spatial hourglass with causal temporal
# attention and no temporal resampling. Encoder blocks run spatial→temporal;
# decoder blocks run temporal→spatial. The bottleneck is factorized or joint 3D.
# Based on https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py.


def _zero_out(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.data.zero_()


def _scale_qk(q: torch.Tensor, k: torch.Tensor, log_scale: torch.Tensor, eps: float):
    scale = log_scale.clamp(max=math.log(100)).exp().sqrt()
    q = (F.normalize(q, p=2, dim=-1, eps=eps) * scale).to(q.dtype)
    k = (F.normalize(k, p=2, dim=-1, eps=eps) * scale).to(k.dtype)
    return q, k


def _validate_inference_options(
    *,
    temporal_resample: list[bool],
    num_levels: int,
    positional_embedding: str,
    temporal_mixer: str,
    dropout: float,
    bottleneck_3d: bool,
    bottleneck_3d_causal_time: bool,
    bottleneck_3d_rope: bool,
) -> None:
    if len(temporal_resample) != num_levels or any(temporal_resample):
        raise ValueError("LiDAR inference requires temporal resampling disabled at every level.")
    if positional_embedding != "learnable_embedding":
        raise ValueError("LiDAR inference requires positional_embedding='learnable_embedding'.")
    if temporal_mixer != "attention":
        raise ValueError("LiDAR inference requires temporal_mixer='attention'.")
    if bottleneck_3d and not (bottleneck_3d_causal_time and bottleneck_3d_rope):
        raise ValueError("LiDAR 3D bottlenecks require causal_time=True and use_3d_rope=True.")
    if not 0 <= dropout <= 1:
        raise ValueError("dropout probability must be between 0 and 1.")


# =============================================================================
# Rotary positional encoding (spatial)
# =============================================================================


class AxialRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_harmonics: list[int]):
        super().__init__()
        freqs_h = self.setup_freqs(num_heads * dim // 4, max_harmonics[0])
        freqs_w = self.setup_freqs(num_heads * dim // 4, max_harmonics[1])
        self.register_buffer("freqs_h", freqs_h.view(dim // 4, num_heads).T)
        self.register_buffer("freqs_w", freqs_w.view(dim // 4, num_heads).T)

    def setup_freqs(self, dim: int, max_harmonics: int):
        return torch.linspace(math.log(1), math.log(max_harmonics), dim).exp().round()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = einops.rearrange(coords, "b c h w -> b h w c")
        radian_h = coords[..., None, [0]] * self.freqs_h
        radian_w = coords[..., None, [1]] * self.freqs_w
        return torch.cat((radian_h, radian_w), dim=-1)

    @staticmethod
    def rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * theta.cos() - x2 * theta.sin()
        y2 = x1 * theta.sin() + x2 * theta.cos()
        return torch.cat((y1, y2), dim=-1)

    def extra_repr(self) -> str:
        return f"freqs_h={tuple(self.freqs_h.shape)}, freqs_w={tuple(self.freqs_w.shape)}"


# =============================================================================
# Spatial self-attention blocks
# =============================================================================


class GlobalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_max_harmonics: list[int] = (1, 1),
        bias=False,
        eps=1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.eps = eps

        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.full([self.num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.rope = AxialRoPE(self.head_dim, num_heads, rope_max_harmonics)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(_zero_out)

    def apply_rope_qk(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        theta = self.rope(coords)
        d = theta.shape[-1] * 2
        assert (q.shape[-1] >= d) and (k.shape[-1] >= d)
        q[..., :d] = self.rope.rotate(q[..., :d], theta).to(q.dtype)
        k[..., :d] = self.rope.rotate(k[..., :d], theta).to(k.dtype)
        return q, k

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        q, k, v = einops.rearrange(qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim)
        q, k = _scale_qk(q, k, self.scale, self.eps)
        q, k = self.apply_rope_qk(q, k, coords)
        q = einops.rearrange(q, "B H W N D -> B N (H W) D")
        k = einops.rearrange(k, "B H W N D -> B N (H W) D")
        v = einops.rearrange(v, "B H W N D -> B N (H W) D")
        h = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        h = einops.rearrange(h, "B N (H W) D -> B H W (N D)", H=H, W=W)
        h = self.out_proj(h)
        return h

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"


class CircularNeighborhoodSelfAttentionBlock(GlobalSelfAttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: list[int],
        dilation: list[int] = 1,
        rope_max_harmonics: list[int] = (1, 1),
        # When True (default), the W (last spatial) axis is treated as
        # periodic: it is circularly padded before the neighborhood attention
        # so the seam wraps (range-image azimuth, column 0 ~ column W-1). When
        # False, no wrap -- the neighborhood shifts inward at the W edges
        # (the same boundary handling it already uses for H). Set False for a
        # Cartesian grid (e.g. BEV x-axis) where the two W edges are unrelated.
        circular: bool = True,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            rope_max_harmonics=rope_max_harmonics,
        )
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.circular = circular

    def before_attn(self, q, k, v):
        if not self.circular:
            return q, k, v
        # TODO: A dilated window needs dilation * (kernel // 2) wrapped columns,
        # so for dilation > 1 the azimuth seam does not fully wrap (k=3, d=2:
        # column 0 attends {0, 2, 4}, not {W-2, 0, 2}). This matches the NATTEN
        # reference the checkpoint was trained with, and shipped configs use
        # dilation 1, where the two agree. Fix together with the reference and
        # retrain; also update after_attn and the oracle tests.
        padding = self.kernel_size[1] // 2
        if padding == 0:
            return q, k, v
        q = F.pad(q, (0, 0, 0, 0, padding, padding), mode="circular")
        k = F.pad(k, (0, 0, 0, 0, padding, padding), mode="circular")
        v = F.pad(v, (0, 0, 0, 0, padding, padding), mode="circular")
        return q, k, v

    def after_attn(self, x):
        if not self.circular:
            return x
        padding = self.kernel_size[1] // 2
        if padding == 0:
            return x
        x = x[:, :, padding:-padding]
        return x

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        q, k, v = einops.rearrange(qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim)
        q, k = _scale_qk(q, k, self.scale, self.eps)
        q, k = self.apply_rope_qk(q, k, coords)
        q, k, v = self.before_attn(q, k, v)
        h = neighborhood_attention_2d(
            query=q,
            key=k,
            value=v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            scale=1.0,
        )
        h = einops.rearrange(h, "B H W N D -> B H W (N D)")
        h = self.after_attn(h)
        h = self.out_proj(h)
        return h

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            f"kernel_size={self.kernel_size}, dilation={self.dilation}"
        )


# =============================================================================
# Causal temporal self-attention
# =============================================================================


class CausalTemporalAttention(nn.Module):
    """Causal self-attention along the temporal dimension.

    Input/output shape: (N, T, C) where N = B*H*W.
    Each position attends only to itself and earlier positions.
    """

    def __init__(self, dim: int, num_heads: int, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(_zero_out)
        self._SDPA_MAX_BATCH = 65535

    def _project_qkv(self, x: torch.Tensor):
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        return einops.rearrange(
            qkv,
            "N T (three heads D) -> three N heads T D",
            three=3,
            D=self.head_dim,
        )

    def _sdpa(self, q, k, v, *, attn_mask=None, is_causal=False):
        N = q.shape[0]
        if N <= self._SDPA_MAX_BATCH:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        chunks = []
        for i in range(0, N, self._SDPA_MAX_BATCH):
            j = min(i + self._SDPA_MAX_BATCH, N)
            chunks.append(
                F.scaled_dot_product_attention(
                    q[i:j],
                    k[i:j],
                    v[i:j],
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )
            )
        return torch.cat(chunks, dim=0)

    def _finish(self, x, h):
        h = einops.rearrange(h, "N heads T D -> N T (heads D)")
        h = self.out_proj(h)
        return x + h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        return self._finish(x, self._sdpa(q, k, v, is_causal=True))

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend new temporal tokens to cached history and this causal block."""
        q, k_new, v_new = self._project_qkv(x)
        past = 0
        if kv_cache is not None:
            k_past, v_past = kv_cache
            if (
                k_past.shape[:2] != k_new.shape[:2]
                or k_past.shape[3:] != k_new.shape[3:]
                or v_past.shape != k_past.shape
            ):
                raise ValueError(
                    "temporal KV cache shape mismatch: "
                    f"cached k/v={tuple(k_past.shape)}/{tuple(v_past.shape)}, "
                    f"new k={tuple(k_new.shape)}"
                )
            past = k_past.shape[2]
            k = torch.cat([k_past, k_new], dim=2)
            v = torch.cat([v_past, v_new], dim=2)
        else:
            k, v = k_new, v_new

        T_new = q.shape[2]
        q_index = past + torch.arange(T_new, device=q.device)[:, None]
        k_index = torch.arange(past + T_new, device=q.device)[None, :]
        causal_offset_mask = k_index <= q_index
        out = self._finish(
            x,
            self._sdpa(
                q,
                k,
                v,
                attn_mask=causal_offset_mask,
                is_causal=False,
            ),
        )
        return out, (k.detach(), v.detach())

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"


# =============================================================================
# Feed-forward
# =============================================================================


class GEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features * 2, bias=bias)

    def forward(self, x):
        h = super().forward(x)
        h, gate = h.chunk(2, dim=-1)
        return h * F.gelu(gate)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gegelu = GEGLU(dim, mid_dim, bias=False)
        self.linear = nn.Linear(mid_dim, dim, bias=False).apply(_zero_out)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gegelu(x)
        x = self.linear(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


# =============================================================================
# Spatial block (spatial attention + FFN)
# =============================================================================


class SpatialBlock(nn.Module):
    """Spatial attention + FFN.  Operates on (BT, H, W, C) per-frame."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        attn_type: Literal["local", "global"] = "global",
        kernel_size: list[int] | None = None,
        dilation: list[int] = 1,
        rope_max_harmonics: list[int] = (1, 1),
        mlp_ratio: float = 3.0,
        # Forwarded to the local (neighborhood) attention: whether the W axis
        # wraps (periodic azimuth) or clamps (Cartesian). No effect for global.
        circular: bool = True,
    ):
        super().__init__()

        if attn_type == "global":
            self.residual_attn = GlobalSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                rope_max_harmonics=rope_max_harmonics,
            )
        elif attn_type == "local":
            self.residual_attn = CircularNeighborhoodSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                rope_max_harmonics=rope_max_harmonics,
                circular=circular,
            )

        self.residual_ffn = FeedForwardNetwork(
            dim=in_dim,
            mid_dim=int(in_dim * mlp_ratio),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.residual_attn(x, coords)
        x = self.residual_ffn(x)
        return x


class TemporalBlock(nn.Module):
    """Causal temporal attention + FFN.  Operates on (BHW, T, C)."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        self.temporal_attn = CausalTemporalAttention(in_dim, num_heads)
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_attn(x)
        x = self.ffn(x)
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, kv_cache = self.temporal_attn.forward_stream(x, kv_cache)
        x = self.ffn(x)
        return x, kv_cache


class Joint3DSelfAttention(nn.Module):
    """Causal joint attention over (T, H, W), with 3D rotary positions."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_t: int,
        len_h: int,
        len_w: int,
        bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.max_t = max_t
        self.max_h = len_h
        self.max_w = len_w

        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.full([num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.rope3d = VideoRopePosition3DEmb(
            head_dim=self.head_dim,
            len_t=max_t,
            len_h=len_h,
            len_w=len_w,
        )
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(_zero_out)

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        if T > self.max_t:
            raise ValueError(
                f"Joint3DSelfAttention received T={T} but was constructed with "
                f"max_t={self.max_t}. Increase bottleneck_3d_max_t."
            )
        if H > self.max_h or W > self.max_w:
            raise ValueError(
                f"Joint3DSelfAttention received (H, W)=({H}, {W}) but the 3D "
                f"RoPE was sized for (len_h, len_w)=({self.max_h}, {self.max_w})."
            )
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        qkv = einops.rearrange(
            qkv,
            "B T H W (three N D) -> three B (T H W) N D",
            three=3,
            D=self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _scale_qk(q, k, self.scale, self.eps)
        rope_emb = self.rope3d(T, H, W)
        q = apply_rotary_emb(q, rope_emb)
        k = apply_rotary_emb(k, rope_emb)
        q = einops.rearrange(q, "B S N D -> B N S D")
        k = einops.rearrange(k, "B S N D -> B N S D")
        v = einops.rearrange(v, "B S N D -> B N S D")

        S = H * W
        t_idx = torch.arange(T, device=q.device).repeat_interleave(S)
        mask = t_idx.unsqueeze(0) <= t_idx.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        out = einops.rearrange(out, "B N (T H W) D -> B T H W (N D)", T=T, H=H, W=W)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend new frames to a sliding cache of unrotated (B, N, T, H*W, D)
        keys and values. Positions are reindexed when the window exceeds max_t.
        """
        B, T_new, H, W, C = x.shape
        if T_new > self.max_t:
            raise ValueError(
                f"streaming chunk T={T_new} exceeds bottleneck_3d_max_t={self.max_t}; use a smaller chunk_frames"
            )
        del coords  # 3D RoPE path does not use polar coords
        if H > self.max_h or W > self.max_w:
            raise ValueError(
                f"Joint3DSelfAttention received (H, W)=({H}, {W}) but the 3D "
                f"RoPE was sized for (len_h, len_w)=({self.max_h}, {self.max_w})."
            )

        S = H * W
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        qkv = einops.rearrange(
            qkv,
            "B T H W (three N D) -> three B (T H W) N D",
            three=3,
            D=self.head_dim,
        )
        q_flat, k_flat, v_flat = qkv[0], qkv[1], qkv[2]
        q_flat, k_flat = _scale_qk(q_flat, k_flat, self.scale, self.eps)
        # Cache / stream in (B, N, T, S, D) so frames can be trimmed.
        q_new = einops.rearrange(q_flat, "B (T S) N D -> B N T S D", T=T_new, S=S)
        k_new = einops.rearrange(k_flat, "B (T S) N D -> B N T S D", T=T_new, S=S)
        v_new = einops.rearrange(v_flat, "B (T S) N D -> B N T S D", T=T_new, S=S)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k_all = torch.cat([k_past, k_new], dim=2)
            v_all = torch.cat([v_past, v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new

        T_total = k_all.shape[2]
        if T_total > self.max_t:
            keep = self.max_t
            k_all = k_all[:, :, -keep:].contiguous()
            v_all = v_all[:, :, -keep:].contiguous()
            T_total = keep

        # Window-local temporal positions 0 .. T_total-1 (matches one-shot when
        # the stream never exceeds max_t).
        rope_full = self.rope3d(T_total, H, W)  # (T_total*S, D)
        rope_full = einops.rearrange(rope_full, "(T S) D -> T S D", T=T_total, S=S)
        rope_q = rope_full[-T_new:].reshape(T_new * S, -1)
        rope_k = rope_full.reshape(T_total * S, -1)
        q = einops.rearrange(q_new, "B N T S D -> B (T S) N D")
        k = einops.rearrange(k_all, "B N T S D -> B (T S) N D")
        v = einops.rearrange(v_all, "B N T S D -> B (T S) N D")
        q = apply_rotary_emb(q, rope_q)
        k = apply_rotary_emb(k, rope_k)
        q = einops.rearrange(q, "B S N D -> B N S D")
        k = einops.rearrange(k, "B S N D -> B N S D")
        v = einops.rearrange(v, "B S N D -> B N S D")

        # With one newly appended frame, every cached key is at or before the
        # query frame. Avoiding an explicit all-true mask lets SDPA use its
        # memory-efficient backend; at LiDAR resolution the dense mask alone can
        # otherwise consume several GiB. Multi-frame chunks still need temporal
        # causality between their newly appended frames.
        mask = None
        if T_new > 1:
            t_q = torch.arange(T_total - T_new, T_total, device=q.device).repeat_interleave(S)
            t_k = torch.arange(T_total, device=q.device).repeat_interleave(S)
            mask = t_k.unsqueeze(0) <= t_q.unsqueeze(1)  # (T_new*S, T_total*S)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        out = einops.rearrange(out, "B N (T H W) D -> B T H W (N D)", T=T_new, H=H, W=W)
        out = self.out_proj(out)
        return x + out, (k_all.detach(), v_all.detach())

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            f"rope3d(len_t={self.max_t}, len_h={self.max_h}, len_w={self.max_w})"
        )


class Bottleneck3DBlock(nn.Module):
    """Causal joint 3D attention + FFN on (B, T, H, W, C)."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        max_t: int,
        mlp_ratio: float = 3.0,
        len_h: int = 1,
        len_w: int = 1,
    ):
        super().__init__()
        self.attn = Joint3DSelfAttention(
            dim=in_dim,
            num_heads=num_heads,
            max_t=max_t,
            len_h=len_h,
            len_w=len_w,
        )
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, coords)
        x = self.ffn(x)
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, kv_cache = self.attn.forward_stream(x, coords, kv_cache)
        x = self.ffn(x)
        return x, kv_cache


# =============================================================================
# Spatial resampling
# =============================================================================


class PatchMerging(nn.Sequential):
    """2x2 spatial downsample: (B, 2H, 2W, C) -> (B, H, W, 2C)"""

    def __init__(self, dim: int):
        super().__init__(
            Rearrange("B (H P1) (W P2) C -> B H W (P1 P2 C)", P1=2, P2=2),
            nn.Linear(4 * dim, 2 * dim, bias=False),
        )


class PatchExpanding(nn.Module):
    """2x2 spatial upsample: (B, H, W, C) -> (B, 2H, 2W, C//2)"""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = einops.rearrange(x, "B H W (P1 P2 C) -> B (H P1) (W P2) C", P1=2, P2=2)
        return x


class Tokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: list[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=False,
            ),
            Rearrange("B C H W -> B H W C"),
        )


class Detokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: list[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            RMSNorm(in_channels),
            nn.Linear(
                in_channels,
                out_channels * patch_size[0] * patch_size[1],
                bias=False,
            ),
            Rearrange(
                "B H W (P1 P2 C) -> B C (H P1) (W P2)",
                P1=patch_size[0],
                P2=patch_size[1],
            ),
        )


# =============================================================================
# Positional embedding
# =============================================================================


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, out_dim: int, resolution: list[int]):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, *resolution, out_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self):
        return self.embedding

    def extra_repr(self):
        return f"resolution={tuple(self.embedding.shape[1:3])}, out_dim={self.embedding.shape[3]}"


# =============================================================================
# Encoder
# =============================================================================


class Encoder(nn.Module):
    """Inference-only encoder; exported configs must disable temporal resampling."""

    def __init__(
        self,
        resolution: list[int],
        in_channels: int,
        z_dim: int,
        base_channels: int = 128,
        patch_size: list[int] = (1, 4),
        window_size: list[int] = (3, 9),
        depths: list[int] = (3, 3, 3, 3),
        num_heads: list[int] = (2, 4, 8, 16),
        dilation: list[int] = (1, 1, 1, 1),
        temporal_downsample: list[bool] = (True, False, False),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: str = "learnable_embedding",
        temporal_first_frame_special: bool = False,
        bottleneck_3d: bool = False,
        bottleneck_3d_max_t: int = 32,
        bottleneck_3d_causal_time: bool = False,
        bottleneck_3d_rope: bool = False,
        circular_padding: bool = True,
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
    ):
        super().__init__()
        _validate_inference_options(
            temporal_resample=temporal_downsample,
            num_levels=len(depths) - 1,
            positional_embedding=positional_embedding,
            temporal_mixer=temporal_mixer,
            dropout=dropout,
            bottleneck_3d=bottleneck_3d,
            bottleneck_3d_causal_time=bottleneck_3d_causal_time,
            bottleneck_3d_rope=bottleneck_3d_rope,
        )
        # mapping_depth, dropout, temporal_conv_kernel and temporal-resampling
        # variant flags remain accepted for exported-config compatibility.
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.depths = depths
        self.bottleneck_3d = bottleneck_3d

        token_size = torch.tensor(self.resolution) // torch.tensor(self.patch_size)
        max_harmonics = (token_size / 2).int()

        self.tokenizer = Tokenizer(in_channels, base_channels, patch_size)

        self.spatial_pe = LearnablePositionalEmbedding(
            out_dim=base_channels,
            resolution=token_size.tolist(),
        )

        # Down levels: spatial blocks + temporal blocks + merging
        # Skip temporal attention at level 0 (highest resolution) for efficiency.
        self.down_levels = nn.ModuleDict()
        for i, num_blocks in enumerate(depths[:-1]):
            dim_i = base_channels << i
            spatial_blocks = nn.ModuleList()
            for j in range(num_blocks):
                spatial_blocks.append(
                    SpatialBlock(
                        in_dim=dim_i,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.down_levels[f"spatial_{i}"] = spatial_blocks
            if i > 0:
                temporal_blocks = nn.ModuleList()
                for j in range(num_blocks):
                    temporal_blocks.append(
                        TemporalBlock(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                        )
                    )
                self.down_levels[f"temporal_{i}"] = temporal_blocks
            self.down_levels[f"merge_{i}"] = PatchMerging(dim_i)

        # Bottleneck: either factorized (global spatial + temporal), or a
        # single joint 3D self-attention over (T*H*W) tokens.
        i = len(depths) - 1
        bottleneck_dim = base_channels << i
        bottleneck_size = (token_size >> i).tolist()
        if bottleneck_3d:
            self.mid_3d = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_3d.append(
                    Bottleneck3DBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        max_t=bottleneck_3d_max_t,
                        mlp_ratio=mlp_ratio,
                        len_h=bottleneck_size[0],
                        len_w=bottleneck_size[1],
                    )
                )
        else:
            self.mid_spatial = nn.ModuleList()
            self.mid_temporal = nn.ModuleList()
            for j in range(depths[-1]):
                self.mid_spatial.append(
                    SpatialBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        attn_type="global",
                        mlp_ratio=mlp_ratio,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                    )
                )
                self.mid_temporal.append(
                    TemporalBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                    )
                )

        # Project to z_dim * 2 (mean + logvar)
        self.head = nn.Sequential(
            RMSNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, z_dim * 2, bias=False),
            Rearrange("B H W C -> B C H W"),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Encode (B, C, T, H, W) pixels into (B, 2*z_dim, T, H_z, W_z)."""
        return self._forward(x, coords, streaming=False)[0]

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Encode new frames, attending to the cached temporal prefix."""
        return self._forward(x, coords, temporal_kv_cache, streaming=True)

    def _forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        streaming: bool,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        B, C, T, H, W = x.shape
        old_cache = temporal_kv_cache or {}
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        c = F.avg_pool2d(coords, kernel_size=self.patch_size, stride=self.patch_size)

        x = einops.rearrange(x, "B C T H W -> (B T) C H W")
        h = self.tokenizer(x) + self.spatial_pe()
        h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T)

        T_cur = T
        for i in range(len(self.depths) - 1):
            _, _, Hc, Wc, _ = h.shape
            has_temporal = f"temporal_{i}" in self.down_levels
            t_blocks = (
                self.down_levels[f"temporal_{i}"] if has_temporal else [None] * len(self.down_levels[f"spatial_{i}"])
            )
            for block_idx, (s_block, t_block) in enumerate(zip(self.down_levels[f"spatial_{i}"], t_blocks)):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c)
                if t_block is not None:
                    h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                    if streaming:
                        cache_key = f"down_levels.temporal_{i}.{block_idx}"
                        h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                    else:
                        h = t_block(h)
                    h = einops.rearrange(h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc)
                else:
                    h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

            h = einops.rearrange(h, "B T H W C -> (B T) H W C")
            h = self.down_levels[f"merge_{i}"](h)
            h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)
            c = F.avg_pool2d(c, kernel_size=2, stride=2)

        _, _, Hc, Wc, _ = h.shape
        if self.bottleneck_3d:
            for block_idx, block in enumerate(self.mid_3d):
                if streaming:
                    cache_key = f"mid_3d.{block_idx}"
                    h, new_cache[cache_key] = block.forward_stream(h, c, old_cache.pop(cache_key, None))
                else:
                    h = block(h, c)
        else:
            for block_idx, (s_block, t_block) in enumerate(zip(self.mid_spatial, self.mid_temporal)):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c)
                h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                if streaming:
                    cache_key = f"mid_temporal.{block_idx}"
                    h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                else:
                    h = t_block(h)
                h = einops.rearrange(h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc)

        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.head(h)
        h = einops.rearrange(h, "(B T) C H W -> B C T H W", B=B, T=T_cur)
        missing = set(old_cache) - set(new_cache)
        if missing:
            raise ValueError(f"unused temporal KV cache keys: {sorted(missing)}")
        return h, new_cache


class Decoder(nn.Module):
    """Inference-only decoder with spatial upsampling and causal temporal mixing."""

    def __init__(
        self,
        resolution: list[int],
        out_channels: int,
        z_dim: int,
        base_channels: int = 128,
        patch_size: list[int] = (1, 4),
        window_size: list[int] = (3, 9),
        depths: list[int] = (3, 3, 3, 3),
        num_heads: list[int] = (2, 4, 8, 16),
        dilation: list[int] = (1, 1, 1, 1),
        temporal_downsample: list[bool] = (True, False, False),
        temporal_upsample: list[bool] | None = None,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: str = "learnable_embedding",
        stem_patchify: bool = False,
        bottleneck_3d: bool = False,
        bottleneck_3d_max_t: int = 32,
        bottleneck_3d_causal_time: bool = False,
        bottleneck_3d_rope: bool = False,
        temporal_expand_wan_style: bool = False,
        out_patch_size: list[int] | None = None,
        circular_padding: bool = True,
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
    ):
        super().__init__()
        _validate_inference_options(
            temporal_resample=temporal_upsample if temporal_upsample is not None else temporal_downsample,
            num_levels=len(depths) - 1,
            positional_embedding=positional_embedding,
            temporal_mixer=temporal_mixer,
            dropout=dropout,
            bottleneck_3d=bottleneck_3d,
            bottleneck_3d_causal_time=bottleneck_3d_causal_time,
            bottleneck_3d_rope=bottleneck_3d_rope,
        )
        if stem_patchify:
            raise ValueError("LiDAR inference does not support the external-encoder patchifying stem.")
        # mapping_depth, dropout, temporal_conv_kernel and temporal-resampling
        # variant flags remain accepted for exported-config compatibility.
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.out_patch_size = _pair(out_patch_size) if out_patch_size is not None else self.patch_size
        self.depths = depths
        self.bottleneck_3d = bottleneck_3d

        # Attention runs on the output-patch grid before final unpatchifying.
        token_size = torch.tensor(self.resolution) // torch.tensor(self.out_patch_size)
        max_harmonics = (token_size / 2).int()
        n_down = len(depths) - 1
        bottleneck_dim = base_channels << n_down
        bottleneck_size = (token_size >> n_down).tolist()

        self.stem = nn.Sequential(
            Rearrange("B C H W -> B H W C"),
            nn.Linear(z_dim, bottleneck_dim, bias=False),
        )

        # Positional embedding at bottleneck resolution
        self.spatial_pe = LearnablePositionalEmbedding(
            out_dim=bottleneck_dim,
            resolution=bottleneck_size,
        )

        # Bottleneck: temporal→spatial, or joint causal 3D attention.
        if bottleneck_3d:
            self.mid_3d = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_3d.append(
                    Bottleneck3DBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        max_t=bottleneck_3d_max_t,
                        mlp_ratio=mlp_ratio,
                        len_h=bottleneck_size[0],
                        len_w=bottleneck_size[1],
                    )
                )
        else:
            self.mid_temporal = nn.ModuleList()
            self.mid_spatial = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_temporal.append(
                    TemporalBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                    )
                )
                self.mid_spatial.append(
                    SpatialBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        attn_type="global",
                        mlp_ratio=mlp_ratio,
                        rope_max_harmonics=(max_harmonics >> n_down).clamp(min=1),
                    )
                )

        # Up levels: spatial expand + temporal→spatial blocks
        # Skip temporal attention at level 0 (highest resolution) for efficiency.
        self.up_levels = nn.ModuleDict()
        for i in reversed(range(n_down)):
            dim_i = base_channels << i
            dim_above = base_channels << (i + 1)

            self.up_levels[f"expand_{i}"] = PatchExpanding(dim_above)

            spatial_blocks = nn.ModuleList()
            for j in range(depths[i]):
                spatial_blocks.append(
                    SpatialBlock(
                        in_dim=dim_i,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.up_levels[f"spatial_{i}"] = spatial_blocks
            if i > 0:
                temporal_blocks = nn.ModuleList()
                for j in range(depths[i]):
                    temporal_blocks.append(
                        TemporalBlock(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                        )
                    )
                self.up_levels[f"temporal_{i}"] = temporal_blocks

        self.detokenizer = Detokenizer(base_channels, out_channels, self.out_patch_size)

    def forward(
        self,
        z: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_temporal_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            z: (B, z_dim, T', H_z, W_z) latent
            coords: (1, 2, H, W) polar coordinates
        Returns:
            (B, C, T, H, W) reconstructed video
        """
        B = z.shape[0]
        T_cur = z.shape[2]
        n_down = len(self.depths) - 1
        streaming = return_temporal_kv_cache
        if temporal_kv_cache is not None and not streaming:
            raise ValueError("temporal_kv_cache requires return_temporal_kv_cache=True")
        old_cache = temporal_kv_cache or {}
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Build coordinate pyramid. The base sits at the *top up-level* token
        # grid (= resolution // out_patch_size), which is the highest-res grid
        # the attention blocks actually see -- the final detokenizer's
        # pixel-shuffle never gets coords.
        c_base = F.avg_pool2d(coords, kernel_size=self.out_patch_size, stride=self.out_patch_size)
        c_levels = [c_base]
        for i in range(n_down):
            c_levels.append(F.avg_pool2d(c_levels[-1], kernel_size=2, stride=2))

        # Stem per-frame + spatial PE
        z = einops.rearrange(z, "B C T H W -> (B T) C H W")
        h = self.stem(z) + self.spatial_pe()  # (BT, H_z, W_z, C)
        h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Bottleneck
        _, _, Hc, Wc, _ = h.shape
        if self.bottleneck_3d:
            for block_idx, block in enumerate(self.mid_3d):
                if streaming:
                    cache_key = f"mid_3d.{block_idx}"
                    h, new_cache[cache_key] = block.forward_stream(h, c_levels[n_down], old_cache.pop(cache_key, None))
                else:
                    h = block(h, c_levels[n_down])
        else:
            # Factorized: temporal then global-spatial (decoder order).
            for block_idx, (t_block, s_block) in enumerate(zip(self.mid_temporal, self.mid_spatial)):
                h = einops.rearrange(h, "B T H W C -> (B H W) T C")
                if streaming:
                    cache_key = f"mid_temporal.{block_idx}"
                    h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                else:
                    h = t_block(h)
                h = einops.rearrange(h, "(B H W) T C -> (B T) H W C", B=B, H=Hc, W=Wc)
                h = s_block(h, c_levels[n_down])
                h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Up levels
        for i in reversed(range(n_down)):
            # Spatial upsample per-frame
            h = einops.rearrange(h, "B T H W C -> (B T) H W C")
            h = self.up_levels[f"expand_{i}"](h)
            h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

            _, _, Hc, Wc, _ = h.shape

            has_temporal = f"temporal_{i}" in self.up_levels
            t_blocks = self.up_levels[f"temporal_{i}"] if has_temporal else [None] * len(self.up_levels[f"spatial_{i}"])
            for block_idx, (t_block, s_block) in enumerate(zip(t_blocks, self.up_levels[f"spatial_{i}"])):
                if t_block is not None:
                    h = einops.rearrange(h, "B T H W C -> (B H W) T C")
                    if streaming:
                        cache_key = f"up_levels.temporal_{i}.{block_idx}"
                        h, new_cache[cache_key] = t_block.forward_stream(h, old_cache.pop(cache_key, None))
                    else:
                        h = t_block(h)
                    h = einops.rearrange(h, "(B H W) T C -> (B T) H W C", B=B, H=Hc, W=Wc)
                else:
                    h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                h = s_block(h, c_levels[i])
                h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Detokenize per-frame
        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.detokenizer(h)  # (BT, C_out, H, W)
        h = einops.rearrange(h, "(B T) C H W -> B C T H W", B=B, T=T_cur)
        if streaming:
            missing = set(old_cache) - set(new_cache)
            if missing:
                raise ValueError(f"unused temporal KV cache keys: {sorted(missing)}")
            return h, new_cache
        return h


# =============================================================================
# LiDAR configuration, checkpoint loading, and inference
# =============================================================================


_LIDAR_VAE_SUBFOLDER = "lidar_vae"
_LIDAR_CONFIG_FIELDS = frozenset(
    {
        "dtype",
        "sample_posterior",
        "apply_validity_mask",
        "fps",
        "latent_channels",
        "temporal_compression_factor",
        "spatial_compression",
        "network_config",
        "range_projection",
        "streaming_chunk_frames",
        "streaming_context_frames",
    }
)


def validate_lidar_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("LiDAR metadata must be a JSON object.")
    if missing := _LIDAR_CONFIG_FIELDS - config.keys():
        raise ValueError(f"Incomplete joint artifact: missing LiDAR metadata {sorted(missing)}.")
    if config["dtype"] != "float32" or config["sample_posterior"] is not False:
        raise ValueError("LiDAR requires FP32 execution and posterior-mean encoding.")
    if not isinstance(config["apply_validity_mask"], bool):
        raise ValueError("LiDAR apply_validity_mask must be boolean.")
    projection = config["range_projection"]
    if not isinstance(projection, dict) or not isinstance(config["network_config"], dict):
        raise ValueError("LiDAR range_projection and network_config must be JSON objects.")
    expected = {
        "semantic_width": 1800,
        "model_width": 1808,
        "native_height": 128,
        "model_width_transform": "circular_pad",
        "intensity_encoding": "unit",
    }
    if any(projection.get(key) != value for key, value in expected.items()):
        raise ValueError("LiDAR requires the 128x1800 metric/unit-intensity grid with circular padding to 1808.")
    minimum, maximum = projection.get("min_range_m"), projection.get("max_range_m")
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in (minimum, maximum)) or maximum <= minimum:
        raise ValueError("LiDAR range normalization requires finite min_range_m < max_range_m.")
    network = config["network_config"]
    if (
        network.get("resolution") != [128, 1808]
        or network.get("patch_size") != [2, 2]
        or len(network.get("depths", [])) != 4
        or config["spatial_compression"] != [16, 16]
        or config["temporal_compression_factor"] != 1
        or network.get("z_dim") != config["latent_channels"]
        or network.get("in_channels") != 3
        or any(network.get("temporal_downsample", [True]))
    ):
        raise ValueError("LiDAR architecture and compression metadata disagree.")
    if not math.isfinite(config["fps"]) or config["fps"] <= 0:
        raise ValueError("LiDAR FPS must be finite and positive.")
    chunk, context = config["streaming_chunk_frames"], config["streaming_context_frames"]
    if type(chunk) is not int or chunk < 1 or (context is not None and (type(context) is not int or context < chunk)):
        raise ValueError("LiDAR streaming context must be at least the positive chunk length.")


def _validate_lidar_vae_config(component: dict[str, Any], deployment: dict[str, Any]) -> None:
    validate_lidar_config(deployment)
    validate_lidar_config(component)
    for key in _LIDAR_CONFIG_FIELDS:
        expected = deployment[key]
        if key == "network_config":
            # The VAE resolves constructor defaults that transformer metadata may omit.
            if any(name not in component[key] or component[key][name] != value for name, value in expected.items()):
                raise ValueError("LiDAR VAE network_config metadata disagrees with transformer deployment metadata.")
        elif key not in component or component[key] != expected:
            raise ValueError(f"LiDAR VAE {key} metadata disagrees with transformer deployment metadata.")


def prepare_lidar_encoder_input(frames: torch.Tensor, projection: dict[str, Any]) -> torch.Tensor:
    """Reference physical normalization after symmetric circular width padding."""
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)
    height, width = projection["native_height"], projection["semantic_width"]
    if (
        frames.ndim != 5
        or frames.shape[1] != 3
        or tuple(frames.shape[-2:]) != (height, width)
        or frames.shape[0] < 1
        or frames.shape[2] < 1
    ):
        raise ValueError(
            f"Expected nonempty LiDAR frames [3,T,{height},{width}] or [B,3,T,{height},{width}], "
            f"got {tuple(frames.shape)}."
        )
    frames = frames.float()
    padding = projection["model_width"] - projection["semantic_width"]
    if padding < 0 or padding % 2:
        raise ValueError("LiDAR model width must allow symmetric circular padding of the semantic width.")
    half_padding = padding // 2
    if half_padding:
        frames = torch.cat((frames[..., -half_padding:], frames, frames[..., :half_padding]), dim=-1)
    minimum, maximum = projection["min_range_m"], projection["max_range_m"]
    ranges, intensities, validity = frames.split(1, dim=1)
    valid = (validity >= 0.5) & (ranges >= minimum) & (ranges <= maximum)
    ranges = (ranges.clamp(minimum, maximum) - minimum) / (maximum - minimum) * 2 - 1
    intensities = intensities.clamp(0, 1) * 2 - 1
    return torch.cat((ranges.masked_fill(~valid, -1), intensities.masked_fill(~valid, -1), valid.float()), dim=1)


class _LidarComponent(nn.Module):
    _unused_prefixes: tuple[str, ...]
    _component_name: str
    weights_sources: list[DiffusersPipelineLoader.ComponentSource]

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: dict[str, Any],
        device: torch.device,
        *,
        load_config: LoadConfig | None = None,
        od_config: OmniDiffusionConfig | None = None,
    ) -> Self:
        """Load a standalone component with the shared loader's configuration.

        Pipeline callers can construct the component from its config, declare
        a ``lidar_vae`` ComponentSource on the pipeline, and route its iterator
        to :meth:`load_weights` without loading or moving weights during init.
        """
        from vllm.config.load import LoadConfig

        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
        from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

        load_config = load_config if load_config is not None else LoadConfig()
        od_config = od_config if od_config is not None else OmniDiffusionConfig(dtype=torch.float32)
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_dir():
            # Fetch metadata only; weight discovery and downloads belong to the
            # shared loader, using the original repository ID and revision.
            checkpoint_path = Path(
                download_weights_from_hf_specific(
                    model_name_or_path=model_path,
                    cache_dir=load_config.download_dir,
                    allow_patterns=[f"{_LIDAR_VAE_SUBFOLDER}/config.json"],
                    revision=od_config.revision,
                    ignore_patterns=load_config.ignore_patterns,
                    require_all=True,
                )
            )
        config_path = checkpoint_path / _LIDAR_VAE_SUBFOLDER / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Incomplete joint artifact: {_LIDAR_VAE_SUBFOLDER}/config.json is required.")
        component_config = json.loads(config_path.read_text())
        _validate_lidar_vae_config(component_config, config)
        model = cls(component_config).float().to(device=device)
        model.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=_LIDAR_VAE_SUBFOLDER,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
            ),
        ]
        DiffusersPipelineLoader(load_config, od_config).load_weights(model)
        return model

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream this component's parameters and buffers from the unified VAE."""
        from vllm.model_executor.models.utils import AutoWeightsLoader

        # Pipeline construction may set the default parameter dtype to BF16.
        # Convert before loading so FP32 checkpoint values are never rounded.
        self.float()
        expected = self.state_dict()
        # Original VAE checkpoints name RMSNorm weights "scale". Derive the
        # mapping from modules so attention temperature scales stay untouched,
        # including norms in Sequential containers (head/detokenizer).
        norm_keys = {
            f"{name}.scale": f"{name}.weight" for name, module in self.named_modules() if isinstance(module, RMSNorm)
        }
        seen: set[str] = set()

        def remapped_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for checkpoint_name, tensor in weights:
                if checkpoint_name.startswith(self._unused_prefixes):
                    continue
                name = norm_keys.get(checkpoint_name, checkpoint_name)
                if name in seen:
                    raise ValueError(f"Duplicate LiDAR {self._component_name} tensor after RMSNorm remapping: {name}.")
                if name not in expected:
                    raise ValueError(f"Unexpected LiDAR {self._component_name} tensors: {[name]}.")
                if expected[name].is_floating_point() and tensor.dtype != torch.float32:
                    raise ValueError(f"LiDAR {self._component_name} tensor {checkpoint_name} must be FP32.")
                if tensor.shape != expected[name].shape:
                    raise RuntimeError(
                        f"LiDAR {self._component_name} size mismatch for {name}: "
                        f"expected {tuple(expected[name].shape)}, got {tuple(tensor.shape)}."
                    )
                seen.add(name)
                yield name, tensor

        loaded = AutoWeightsLoader(self).load_weights(remapped_weights())
        # The shared loader checks parameters; LiDAR also requires every
        # persistent buffer, particularly coordinates and latent statistics.
        if missing := expected.keys() - loaded:
            raise RuntimeError(f"Missing keys in LiDAR {self._component_name} checkpoint: {sorted(missing)}.")
        if (
            not torch.isfinite(self.latent_mean).all()
            or not torch.isfinite(self.latent_std).all()
            or (self.latent_std <= 0).any()
        ):
            raise ValueError("LiDAR latent statistics must be finite with positive standard deviations.")
        self.eval().requires_grad_(False)
        return loaded


class Cosmos3LidarEncoder(_LidarComponent):
    _unused_prefixes = ("decoder.", "post_quant_conv.")
    _component_name = "encoder"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_lidar_config(config)

        self.config = config
        network = config["network_config"]
        encoder_args = set(inspect.signature(Encoder).parameters)
        self.encoder = Encoder(**{key: value for key, value in network.items() if key in encoder_args})
        channels = config["latent_channels"]
        self.quant_conv = nn.Conv2d(channels * 2, channels * 2, 1)
        self.register_buffer("coords", generate_polar_coords(*network["resolution"]))
        self.register_buffer("latent_mean", torch.empty(1, channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.empty(1, channels, 1, 1, 1))

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Keep normalization, network execution and the latent affine in FP32,
        # including when the parent DiT is under BF16 autocast.
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            state = prepare_lidar_encoder_input(frames.to(self.coords.device), self.config["range_projection"])
            chunk = self.config["streaming_chunk_frames"]
            context = self.config["streaming_context_frames"]
            outputs, cache = [], None
            for start in range(0, state.shape[2], chunk):
                pixels = state[:, :, start : start + chunk]
                if cache is not None and context is not None:
                    keep = context - pixels.shape[2]
                    cache = (
                        {key: (k[:, :, -keep:], v[:, :, -keep:]) for key, (k, v) in cache.items()} if keep > 0 else None
                    )
                parameters, cache = self.encoder.forward_stream(pixels, self.coords, cache)
                batch, _, time, height, width = parameters.shape
                parameters = parameters.permute(0, 2, 1, 3, 4).flatten(0, 1)
                mean = self.quant_conv(parameters).chunk(2, dim=1)[0]
                outputs.append(mean.reshape(batch, time, -1, height, width).permute(0, 2, 1, 3, 4))
            latent = torch.cat(outputs, dim=2)
            return (latent - self.latent_mean) / self.latent_std


def lidar_decoder_args(network: dict[str, Any]) -> dict[str, Any]:
    """Resolve the decoder topology exactly as the reference TransformerVAE."""

    args = {key: value for key, value in network.items() if key in inspect.signature(Decoder).parameters}
    # TransformerVAE never enables the standalone decoder's external-encoder stem.
    args.pop("stem_patchify", None)
    for field in ("depths", "num_heads", "dilation"):
        if network.get(f"decoder_{field}") is not None:
            args[field] = network[f"decoder_{field}"]
    if network.get("decoder_depths") is not None:
        temporal = network.get("decoder_temporal_upsample")
        if temporal is None:
            raise ValueError("An asymmetric LiDAR decoder requires decoder_temporal_upsample.")
        args.update(temporal_downsample=temporal, temporal_upsample=temporal)
    return args


def validate_lidar_decoder_config(config: dict[str, Any]) -> None:
    validate_lidar_config(config)
    network = config["network_config"]
    if (
        network.get("out_channels") != 3
        or network.get("predict_validity", False)
        or network.get("mask_as_input", False)
        or network.get("formulation", "VAE") != "VAE"
    ):
        raise ValueError("LiDAR decoding requires three output channels with embedded validity logits.")
    args = lidar_decoder_args(network)
    depths = args["depths"]
    patch = args.get("out_patch_size") or args["patch_size"]
    temporal = args.get("temporal_upsample")
    if temporal is None:
        temporal = args["temporal_downsample"]
    if (
        len(depths) < 1
        or len(patch) != 2
        or [size * 2 ** (len(depths) - 1) for size in patch] != config["spatial_compression"]
        or len(temporal) != len(depths) - 1
        or any(temporal)
        or any(len(args.get(field, ())) != len(depths) for field in ("num_heads", "dilation"))
    ):
        raise ValueError("LiDAR decoder topology must preserve spatial and temporal compression.")
    if args.get("temporal_mixer", "attention") != "attention" or (
        args.get("bottleneck_3d", False)
        and not (args.get("bottleneck_3d_causal_time", False) and args.get("bottleneck_3d_rope", False))
    ):
        raise ValueError("LiDAR streaming decode requires causal temporal attention with RoPE for a 3D bottleneck.")
    threshold = config["range_projection"].get("validity_threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 < threshold < 1:
        raise ValueError("LiDAR validity_threshold must lie in (0, 1).")


def postprocess_lidar_decoder_output(output: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """Convert raw network predictions to the unpadded metric sensor grid."""
    projection = config["range_projection"]
    if (
        output.ndim != 5
        or output.shape[1] != 3
        or tuple(output.shape[-2:])
        != (
            projection["native_height"],
            projection["model_width"],
        )
    ):
        raise ValueError(f"Unexpected LiDAR decoder output shape: {tuple(output.shape)}.")
    output = output.float()
    minimum, maximum = projection["min_range_m"], projection["max_range_m"]
    ranges = (output[:, :1].clamp(-1, 1) + 1) * 0.5 * (maximum - minimum) + minimum
    intensity = (output[:, 1:2].clamp(-1, 1) + 1) * 0.5
    validity = output[:, 2:3].sigmoid()
    if config["apply_validity_mask"]:
        valid = validity >= projection.get("validity_threshold", 0.5)
        ranges = ranges.masked_fill(~valid, 0)
        intensity = intensity.masked_fill(~valid, 0)
        validity = valid.float()
    frames = torch.cat((ranges, intensity, validity), dim=1)
    offset = (projection["model_width"] - projection["semantic_width"]) // 2
    return frames[..., offset : offset + projection["semantic_width"]].contiguous()


class Cosmos3LidarDecoder(_LidarComponent):
    """Inference-only, FP32 decoder for normalized diffusion latents."""

    _unused_prefixes = ("encoder.", "quant_conv.")
    _component_name = "decoder"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_lidar_decoder_config(config)

        self.config = config
        self.decoder = Decoder(**lidar_decoder_args(config["network_config"]))
        channels = config["latent_channels"]
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)
        self.register_buffer("coords", generate_polar_coords(*config["network_config"]["resolution"]))
        self.register_buffer("latent_mean", torch.empty(1, channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.empty(1, channels, 1, 1, 1))

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return [B,3,T,128,1800] metric frames, keeping each request's cache local."""
        expected_hw = tuple(
            size // factor
            for size, factor in zip(self.config["network_config"]["resolution"], self.config["spatial_compression"])
        )
        if (
            latents.ndim != 5
            or latents.shape[1] != self.config["latent_channels"]
            or tuple(latents.shape[-2:]) != expected_hw
            or latents.shape[0] < 1
            or latents.shape[2] < 1
        ):
            raise ValueError(f"Expected nonempty LiDAR latents [B,{self.config['latent_channels']},T,{expected_hw}].")
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            latents = latents.to(device=self.coords.device, dtype=torch.float32)
            latents = latents * self.latent_std + self.latent_mean
            chunk = self.config["streaming_chunk_frames"]
            context = self.config["streaming_context_frames"]
            outputs, cache = [], None
            for start in range(0, latents.shape[2], chunk):
                latent = latents[:, :, start : start + chunk]
                batch, _, time, height, width = latent.shape
                if cache is not None and context is not None:
                    keep = context - time
                    cache = (
                        {key: (k[:, :, -keep:], v[:, :, -keep:]) for key, (k, v) in cache.items()} if keep > 0 else None
                    )
                latent = latent.permute(0, 2, 1, 3, 4).flatten(0, 1)
                latent = self.post_quant_conv(latent)
                latent = latent.reshape(batch, time, -1, height, width).permute(0, 2, 1, 3, 4)
                output, cache = self.decoder(
                    latent, self.coords, temporal_kv_cache=cache, return_temporal_kv_cache=True
                )
                outputs.append(postprocess_lidar_decoder_output(output, self.config))
            return torch.cat(outputs, dim=2)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)
