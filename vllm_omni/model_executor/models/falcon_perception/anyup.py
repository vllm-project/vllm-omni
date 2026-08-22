# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
"""AnyUp feature upsampler, vendored for Falcon Perception.

This is an adaptation of AnyUp by Thomas Wimmer, Prune Truong, Marie-Julie
Rakotosaona, Michael Oechsle, Federico Tombari, Bernt Schiele, and Jan Eric
Lenssen:

* source: https://github.com/wimmerth/anyup
* license: CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/

It was flattened and modified by Technology Innovation Institute for the
Apache-2.0 Falcon Perception reference implementation
(https://github.com/tiiuae/Falcon-Perception), then adapted for vLLM-Omni.
"""

from functools import lru_cache

import einops as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_COMPILED_FLEX_ATTN = None


def _flex_attn_prefill(*args, **kwargs):
    """Compile ``flex_attention`` on first call rather than at import."""
    global _COMPILED_FLEX_ATTN
    if _COMPILED_FLEX_ATTN is None:
        _COMPILED_FLEX_ATTN = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTN(*args, **kwargs)


# ---------------------------------------------------------------------------
# ResBlock (from layers/convolutions.py)
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super().__init__()
        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else (lambda c: nn.Identity())
        p = kernel_size // 2
        self.block = nn.Sequential(
            N(in_channels),
            activation_fn(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=p,
                padding_mode=pad_mode,
                bias=False,
            ),
            N(out_channels),
            activation_fn(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding=p,
                padding_mode=pad_mode,
                bias=False,
            ),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
            if use_conv_shortcut or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# LearnedFeatureUnification (from layers/feature_unification.py)
# ---------------------------------------------------------------------------


class LearnedFeatureUnification(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        init_gaussian_derivatives: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(b, self.out_channels, c, h, w)
        attn = F.softmax(x, dim=1)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(feats, basis, k):
        b, c, h, w = feats.shape
        p = k // 2
        x = F.pad(feats, (p, p, p, p), value=0)
        x = F.conv2d(x, basis.repeat(c, 1, 1, 1), groups=c)
        mask = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
        denom = F.conv2d(
            F.pad(mask, (p, p, p, p), value=0),
            torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype),
        )
        return x / denom


# ---------------------------------------------------------------------------
# RoPE (from layers/positional_encoding.py) – AnyUp-internal, separate from
# the main model's 3D RoPE
# ---------------------------------------------------------------------------


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class AnyUpRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + _rotate_half(x) * angle.sin()


# ---------------------------------------------------------------------------
# Attention masking (from layers/attention/attention_masking.py)
# ---------------------------------------------------------------------------


def window2d(
    low_res: int | tuple[int, int],
    high_res: int | tuple[int, int],
    ratio: float,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Calculate the lower and upper bounds of row and col for each pixel/position"""
    if isinstance(high_res, int):
        H = W = high_res
    else:
        H, W = high_res
    if isinstance(low_res, int):
        Lh = Lw = low_res
    else:
        Lh, Lw = low_res

    r_pos = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    c_pos = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    pos_r, pos_c = torch.meshgrid(r_pos, c_pos, indexing="ij")

    r_lo = (pos_r - ratio).clamp(0.0, 1.0)
    r_hi = (pos_r + ratio).clamp(0.0, 1.0)
    c_lo = (pos_c - ratio).clamp(0.0, 1.0)
    c_hi = (pos_c + ratio).clamp(0.0, 1.0)

    r0 = (r_lo * Lh).floor().long()
    r1 = (r_hi * Lh).ceil().long()
    c0 = (c_lo * Lw).floor().long()
    c1 = (c_hi * Lw).ceil().long()

    return torch.stack([r0, r1, c0, c1], dim=2)


def get_attention_mask_mod(high_res_h, high_res_w, low_res_h, low_res_w, window_size_ratio=0.1, device="cpu"):
    """Window Attention as above but for FlexAttention."""
    h, w = high_res_h, high_res_w
    h_, w_ = low_res_h, low_res_w

    windows = window2d(
        low_res=(h_, w_),
        high_res=(h, w),
        ratio=window_size_ratio,
        device=device,
    )

    r0 = windows[..., 0]
    r1 = windows[..., 1]
    c0 = windows[..., 2]
    c1 = windows[..., 3]

    def _mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_r_idx = q_idx // w
        q_c_idx = q_idx % w
        kv_r_idx = kv_idx // w_
        kv_c_idx = kv_idx % w_
        row_lower = kv_r_idx >= r0[q_r_idx, q_c_idx]
        row_upper = kv_r_idx < r1[q_r_idx, q_c_idx]
        col_lower = kv_c_idx >= c0[q_r_idx, q_c_idx]
        col_upper = kv_c_idx < c1[q_r_idx, q_c_idx]

        return row_lower & row_upper & col_lower & col_upper

    return _mask_mod


@lru_cache(maxsize=32)
def build_upsampler_block_mask(
    H: int,
    W: int,
    h: int,
    w: int,
    ratio: float = 0.1,
    BLOCK_SIZE: int = 128,
    device: str | torch.device = "cpu",
):
    """Build a FlexAttention ``BlockMask`` for the AnyUp upsampler analytically.
    Equivalent to:
        mask = create_attention_mask(
                get_attn_mask_mod(H, W, h, w, device=device),
                B=None, H=None, Q_LEN=H * W, KV_LEN=h * w,
            )
    Instead of running the expensive compiled ``create_block_mask`` Triton
    kernel (~100 ms), this computes the block-level sparsity pattern directly
    from window geometry using numpy (~1-5 ms) and returns a ``BlockMask``
    via ``BlockMask.from_kv_blocks``.

    Parameters match ``get_attention_mask_mod`` — *H, W* are high-res
    (query) dims, *h, w* are low-res (key/value) dims.
    """

    BQ = BKV = BLOCK_SIZE
    Q_LEN = H * W
    KV_LEN = h * w
    num_q_blocks = (Q_LEN + BQ - 1) // BQ
    num_kv_blocks = (KV_LEN + BKV - 1) // BKV

    # --- per-Q-block union window bounds (O(num_q_blocks), not O(Q_LEN)) ---
    # Q pixels are row-major in the H×W grid.  Each Q block covers a
    # contiguous slice of BQ linear indices.  We find the extreme row/col
    # corners per block and evaluate the window bounds only at those corners.
    first_q = np.arange(num_q_blocks, dtype=np.int64) * BQ
    last_q = np.minimum(first_q + BQ, Q_LEN) - 1

    first_row = first_q // W
    last_row = last_q // W
    first_col = first_q % W
    last_col = last_q % W
    multi_row = first_row < last_row

    # Minimum-row pixel gives the smallest union_r0.
    min_r_pos = (first_row.astype(np.float64) + 0.5) / H
    union_r0 = np.floor(np.clip(min_r_pos - ratio, 0.0, 1.0) * h).astype(np.int64)

    # Maximum-row pixel gives the largest union_r1.
    max_r_pos = (last_row.astype(np.float64) + 0.5) / H
    union_r1 = np.ceil(np.clip(max_r_pos + ratio, 0.0, 1.0) * h).astype(np.int64)

    # For multi-row blocks every column (0..W-1) is present, giving the
    # widest possible column window.  For single-row blocks the column
    # range is [first_col, last_col].
    min_c_col = np.where(multi_row, 0, first_col)
    max_c_col = np.where(multi_row, W - 1, last_col)
    min_c_pos = (min_c_col.astype(np.float64) + 0.5) / W
    max_c_pos = (max_c_col.astype(np.float64) + 0.5) / W
    union_c0 = np.floor(np.clip(min_c_pos - ratio, 0.0, 1.0) * w).astype(np.int64)
    union_c1 = np.ceil(np.clip(max_c_pos + ratio, 0.0, 1.0) * w).astype(np.int64)

    # Conservative KV block range: all blocks between the first and last
    # linear KV index covered by the union window.
    first_kv_lin = union_r0 * w + union_c0
    last_kv_lin = np.maximum((union_r1 - 1) * w + (union_c1 - 1), first_kv_lin)
    first_kv = first_kv_lin // BKV
    last_kv = np.minimum(last_kv_lin // BKV, num_kv_blocks - 1)
    kv_count = last_kv - first_kv + 1

    # Padded Q blocks (beyond Q_LEN) attend to nothing.
    kv_count[first_q >= Q_LEN] = 0

    max_kv = int(kv_count.max()) if kv_count.max() > 0 else 1

    # Build kv_indices: (num_q_blocks, num_kv_blocks).
    # The last dim MUST equal num_kv_blocks because from_kv_blocks /
    # _ordered_to_dense uses shape[-1] as the dense matrix width.
    offsets = np.arange(max_kv, dtype=np.int64)[None, :]
    indices = first_kv[:, None] + offsets
    valid = offsets < kv_count[:, None]
    indices = np.where(valid, indices, 0).astype(np.int32)
    kv_count_i32 = kv_count.astype(np.int32)

    full_indices = np.zeros((num_q_blocks, num_kv_blocks), dtype=np.int32)
    full_indices[:, :max_kv] = indices

    # Tensors with [B=1, H=1, ...] dims, placed on target device.
    kv_num_blocks_t = torch.from_numpy(kv_count_i32).view(1, 1, num_q_blocks).to(device)
    kv_indices_t = torch.from_numpy(full_indices).view(1, 1, num_q_blocks, num_kv_blocks).to(device)

    mask_mod = get_attention_mask_mod(H, W, h, w, ratio, device=str(device))

    return BlockMask.from_kv_blocks(
        kv_num_blocks_t,
        kv_indices_t,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=(BQ, BKV),
        mask_mod=mask_mod,
        seq_lengths=(Q_LEN, KV_LEN),
    )


# ---------------------------------------------------------------------------
# Cross-attention (from layers/attention/chunked_attention.py)
# ---------------------------------------------------------------------------


class AttentionWrapper(nn.Module):
    def __init__(self, qk_dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty([qk_dim * 3, qk_dim]))
        self.in_proj_bias = nn.Parameter(torch.empty([qk_dim * 3]))

    def forward(self, x_q, x_k, x_v):
        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        x_q = x_q @ w_q.T + b_q
        x_k = x_k @ w_k.T + b_k
        return x_q, x_k, x_v


class FlexCrossAttention(nn.Module):
    def __init__(self, qk_dim: int, num_heads: int, **kwargs):
        super().__init__()
        self.dim = qk_dim
        self.num_head = num_heads
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.attention = AttentionWrapper(qk_dim)

    def forward(self, query, key, value, mask=None, **kwargs):
        x_q = self.norm_q(query)
        x_k = self.norm_k(key)
        x_q, x_k, x_v = self.attention(x_q, x_k, value)
        x_q = E.rearrange(x_q, "b HW (h d) -> b h HW d", h=self.num_head)
        x_k = E.rearrange(x_k, "b hw (h d) -> b h hw d", h=self.num_head)

        x_v = E.rearrange(value, "b hw (h d) -> b h hw d", h=self.num_head)
        output = _flex_attn_prefill(x_q, x_k, x_v, block_mask=mask)
        output = E.rearrange(output, "b h hw d -> b hw (h d)")

        return output


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        qk_dim,
        num_heads,
        window_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.cross_attn = FlexCrossAttention(qk_dim, num_heads)
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, block_mask, **kwargs):
        b, _, h, w = q.shape

        q = self.conv2d(q)
        q = E.rearrange(q, "b c h w -> b (h w) c")
        k = E.rearrange(k, "b c h w -> b (h w) c")
        v = E.rearrange(v, "b c h w -> b (h w) c")

        features = self.cross_attn(q, k, v, mask=block_mask)
        return E.rearrange(features, "b (h w) c -> b c h w", h=h, w=w)


# ---------------------------------------------------------------------------
# AnyUp (from model.py)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def _pool_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Reshape-based area pooling that works with symbolic shapes.

    Equivalent to ``F.adaptive_avg_pool2d(x, size)`` when the input spatial
    dims are exactly divisible by the target size.  Uses reshape + mean
    instead of adaptive_avg_pool2d so the inductor can lower it without
    concrete-value guards on kernel size.
    """
    b, c, H, W = x.shape
    oh, ow = size
    if H == oh and W == ow:
        return x
    return x.reshape(b, c, oh, H // oh, ow, W // ow).mean(dim=(3, 5))


def create_coordinate(h, w, start=0.0, end=1.0, device=None, dtype=None):
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx, yy), -1).view(1, h * w, 2)


class AnyUp(nn.Module):
    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
        init_gaussian_derivatives=False,
        **kwargs,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.window_ratio = window_ratio
        self._rb_args = dict(
            kernel_size=1,
            num_groups=8,
            pad_mode="reflect",
            norm_fn=nn.GroupNorm,
            activation_fn=nn.SiLU,
        )

        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(
            None,
            1,
            first_layer_k=kernel_size_lfu,
            init_gaussian_derivatives=init_gaussian_derivatives,
        )

        self.cross_decode = CrossAttentionBlock(qk_dim=qk_dim, num_heads=num_heads, window_ratio=window_ratio)
        self.aggregation = self._make_encoder(2 * qk_dim, 3)

        self.rope = AnyUpRoPE(qk_dim)

        self._compiled = False

    def compile(self, *, mode: str | None = None, dynamic: bool = True):
        if self._compiled:
            return self
        self.forward = torch.compile(self.forward, dynamic=dynamic, mode=mode)
        self._compiled = True
        return self

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0, init_gaussian_derivatives=False):
        pre = (
            nn.Conv2d(
                in_ch,
                self.qk_dim,
                k,
                padding=k // 2,
                padding_mode="reflect",
                bias=False,
            )
            if first_layer_k == 0
            else LearnedFeatureUnification(
                self.qk_dim,
                first_layer_k,
                init_gaussian_derivatives=init_gaussian_derivatives,
            )
        )
        blocks = [ResBlock(self.qk_dim, self.qk_dim, **self._rb_args) for _ in range(layers)]
        return nn.Sequential(pre, *blocks)

    def upsample(self, enc_img, feats, attn_mask, out_size, vis_attn=False, q_chunk_size=None):
        b, c, h, w = feats.shape

        q = _pool_to(self.query_encoder(enc_img), out_size)
        k = _pool_to(self.key_encoder(enc_img), (h, w))
        k = torch.cat([k, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
        k = self.aggregation(k)
        v = feats

        result = self.cross_decode(q, k, v, attn_mask, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
        return result

    def forward(
        self,
        images,
        features,
        attn_mask,
        output_size=None,
        vis_attn=False,
        q_chunk_size=None,
    ):
        output_size = output_size if output_size is not None else images.shape[-2:]
        images = images * 0.5 + 0.5
        images = (images - IMAGENET_MEAN.to(images)) / IMAGENET_STD.to(images)
        images = images.to(features)
        enc = self.image_encoder(images)
        h = enc.shape[-2]
        coords = create_coordinate(h, enc.shape[-1], device=enc.device, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)
        enc = enc.view(enc.shape[0], h, -1, enc.shape[-1]).permute(0, 3, 1, 2)

        result = self.upsample(
            enc,
            features,
            attn_mask,
            output_size,
            vis_attn=vis_attn,
            q_chunk_size=q_chunk_size,
        )
        return result
