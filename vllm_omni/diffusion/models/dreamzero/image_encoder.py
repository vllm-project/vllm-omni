# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero image encoder.

Only the visual tower used by DreamZero I2V inference is ported here. The
checkpoint keys under `action_head.image_encoder.*` load via simple prefix
stripping.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DreamZeroLayerNorm(nn.LayerNorm):
    """LayerNorm that preserves the input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).type_as(x)


class DreamZeroVisionSelfAttention(nn.Module):
    """Self-attention for the vision tower."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        x = self.proj(x)
        return F.dropout(x, self.proj_dropout, self.training)


class DreamZeroVisionAttentionBlock(nn.Module):
    """Attention block for the vision tower."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        num_heads: int,
        post_norm: bool = False,
        activation: str = "gelu",
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if activation != "gelu":
            raise ValueError(f"DreamZero image encoder uses GELU; got activation={activation!r}.")
        self.post_norm = post_norm
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = DreamZeroLayerNorm(dim, eps=norm_eps)
        self.attn = DreamZeroVisionSelfAttention(
            dim,
            num_heads,
            proj_dropout=proj_dropout,
        )
        self.norm2 = DreamZeroLayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DreamZeroVisionTransformer(nn.Module):
    """Vision transformer used by the image encoder."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        mlp_ratio: float = 4.0,
        out_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 32,
        pool_type: str = "token",
        pre_norm: bool = True,
        post_norm: bool = False,
        activation: str = "gelu",
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if pool_type != "token":
            raise ValueError(f"DreamZero image encoder only supports pool_type='token', got {pool_type!r}.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type

        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm,
        )
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            gain * torch.randn(1, self.num_patches + 1, dim),
        )
        self.dropout = nn.Dropout(embedding_dropout)
        self.pre_norm = DreamZeroLayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(
            *[
                DreamZeroVisionAttentionBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    post_norm=post_norm,
                    activation=activation,
                    proj_dropout=proj_dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = DreamZeroLayerNorm(dim, eps=norm_eps)
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x: torch.Tensor, use_31_block: bool = False) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat(
            [
                self.cls_embedding.expand(batch_size, -1, -1).to(dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = self.dropout(x + self.pos_embedding.to(dtype=x.dtype, device=x.device))
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        if use_31_block:
            return self.transformer[:-1](x)
        return self.transformer(x)


class _DreamZeroCLIPContainer(nn.Module):
    """Container matching checkpoint names under `model.visual.*`."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.ones(()))
        self.visual = DreamZeroVisionTransformer(
            image_size=224,
            patch_size=14,
            dim=1280,
            mlp_ratio=4.0,
            out_dim=1024,
            num_heads=16,
            num_layers=32,
            pool_type="token",
            pre_norm=True,
            post_norm=False,
            activation="gelu",
            proj_dropout=0.0,
            embedding_dropout=0.0,
            norm_eps=1e-5,
        )


class DreamZeroImageEncoder(nn.Module):
    """Image encoder wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _DreamZeroCLIPContainer()
        # returns a composed transform whose last stage is CLIP normalization.
        self.transforms = T.Compose(
            [
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def encode_image(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode images for I2V conditioning."""
        size = (self.model.visual.image_size,) * 2
        videos = torch.cat(
            [
                F.interpolate(
                    frame_batch,
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                for frame_batch in videos
            ]
        )
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        param_dtype = next(iter(self.model.visual.parameters())).dtype
        videos = videos.to(dtype=param_dtype)
        out = self.model.visual(videos, use_31_block=True)
        return out.clone()
