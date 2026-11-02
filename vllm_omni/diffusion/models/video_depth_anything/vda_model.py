# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video Depth Anything: a DINOv2 encoder and a DPT head with temporal attention.

Ported from the reference implementation
(https://github.com/DepthAnything/Video-Depth-Anything, Apache-2.0) and kept
numerically equivalent to it. Module and parameter names match the released
``*video_depth_anything_vit{s,b,l}.pth`` checkpoints so they load strictly.
Attention runs through ``F.scaled_dot_product_attention``; the reference's
xFormers and ``baddbmm`` paths compute the same softmax attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PATCH_SIZE = 14
IMAGE_SIZE = 518  # The size the DINOv2 position embeddings were trained at.


@dataclass(frozen=True)
class VideoDepthAnythingConfig:
    """One released checkpoint: encoder size and whether depth is metric."""

    encoder: str
    metric: bool = False

    @property
    def embed_dim(self) -> int:
        return {"vits": 384, "vitb": 768, "vitl": 1024}[self.encoder]

    @property
    def depth(self) -> int:
        return {"vits": 12, "vitb": 12, "vitl": 24}[self.encoder]

    @property
    def num_heads(self) -> int:
        return {"vits": 6, "vitb": 12, "vitl": 16}[self.encoder]

    @property
    def intermediate_layers(self) -> tuple[int, ...]:
        return {"vits": (2, 5, 8, 11), "vitb": (2, 5, 8, 11), "vitl": (4, 11, 17, 23)}[self.encoder]

    @property
    def features(self) -> int:
        return {"vits": 64, "vitb": 128, "vitl": 256}[self.encoder]

    @property
    def out_channels(self) -> tuple[int, int, int, int]:
        return {
            "vits": (48, 96, 192, 384),
            "vitb": (96, 192, 384, 768),
            "vitl": (256, 512, 1024, 1024),
        }[self.encoder]


# ---------------------------------------------------------------------------
# DINOv2 encoder
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, dim))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, 4 * dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    """The DINOv2 backbone as Depth Anything uses it: no registers, layer scale, 518-pixel embeddings."""

    def __init__(self, config: VideoDepthAnythingConfig):
        super().__init__()
        dim = config.embed_dim
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (IMAGE_SIZE // PATCH_SIZE) ** 2 + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))  # Training-only; kept so checkpoints load strictly.
        self.blocks = nn.ModuleList(Block(dim, config.num_heads) for _ in range(config.depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = x.shape[1] - 1
        trained = self.pos_embed.shape[1] - 1
        if num_patches == trained and height == width:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos, patch_pos = pos_embed[:, 0], pos_embed[:, 1:]
        side = math.sqrt(trained)
        # The reference offsets the target grid by 0.1 to avoid a floating-point miss in the scale factor.
        rows, cols = height // PATCH_SIZE + 0.1, width // PATCH_SIZE + 0.1
        patch_pos = F.interpolate(
            patch_pos.reshape(1, int(side), int(side), -1).permute(0, 3, 1, 2),
            scale_factor=(rows / side, cols / side),
            mode="bicubic",
            antialias=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, x.shape[-1])
        return torch.cat((class_pos.unsqueeze(0), patch_pos), dim=1).to(x.dtype)

    def get_intermediate_layers(
        self, x: torch.Tensor, layers: tuple[int, ...]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Normalized (patch tokens, class token) after each requested block."""
        height, width = x.shape[-2:]
        tokens = self.patch_embed(x)
        tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, height, width)
        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in layers:
                normed = self.norm(tokens)
                outputs.append((normed[:, 1:], normed[:, 0]))
        return outputs


# ---------------------------------------------------------------------------
# Temporal motion modules
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class TemporalAttention(nn.Module):
    """Self-attention across frames at every spatial position, with absolute frame positions."""

    def __init__(self, dim: int, heads: int, max_len: int):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.pos_encoder = PositionalEncoding(dim, max_len)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        # [(B F), D, C] -> [(B D), F, C]: attend over the F frames at each of the D positions.
        batch_frames, positions, channels = x.shape
        x = x.reshape(batch_frames // frames, frames, positions, channels).transpose(1, 2)
        x = self.pos_encoder(x.reshape(-1, frames, channels))

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(t.shape[0], frames, self.heads, channels // self.heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(self.to_q(x)), heads(self.to_k(x)), heads(self.to_v(x)))
        out = self.to_out[0](out.transpose(1, 2).reshape(-1, frames, channels))
        out = out.reshape(batch_frames // frames, positions, frames, channels).transpose(1, 2)
        return out.reshape(batch_frames, positions, channels)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.ModuleList([GEGLU(dim, 4 * dim), nn.Dropout(0.0), nn.Linear(4 * dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[0](x))


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, max_len: int):
        super().__init__()
        self.attention_blocks = nn.ModuleList(TemporalAttention(dim, heads, max_len) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(2))
        self.ff = FeedForward(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        for attention, norm in zip(self.attention_blocks, self.norms):
            x = x + attention(norm(x), frames)
        return x + self.ff(self.ff_norm(x))


class TemporalTransformer3DModel(nn.Module):
    def __init__(self, channels: int, heads: int, max_len: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList([TemporalTransformerBlock(channels, heads, max_len)])
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        """``x`` is [(B F), C, H, W]; returns the same shape."""
        residual = x
        batch_frames, channels, height, width = x.shape
        hidden = self.norm(x).permute(0, 2, 3, 1).reshape(batch_frames, height * width, channels)
        hidden = self.proj_in(hidden)
        for block in self.transformer_blocks:
            hidden = block(hidden, frames)
        hidden = self.proj_out(hidden)
        return hidden.reshape(batch_frames, height, width, channels).permute(0, 3, 1, 2) + residual


class TemporalModule(nn.Module):
    def __init__(self, channels: int, max_len: int):
        super().__init__()
        self.temporal_transformer = TemporalTransformer3DModel(channels, heads=8, max_len=max_len)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        return self.temporal_transformer(x, frames)


# ---------------------------------------------------------------------------
# DPT head
# ---------------------------------------------------------------------------


class ResidualConvUnit(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.out_conv = nn.Conv2d(width, width, kernel_size=1)
        self.resConfUnit1 = ResidualConvUnit(width)
        self.resConfUnit2 = ResidualConvUnit(width)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None, size: tuple[int, int] | None = None):
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        modifier = {"scale_factor": 2} if size is None else {"size": size}
        x = F.interpolate(x.contiguous(), **modifier, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTHeadTemporal(nn.Module):
    """DPT decoder with temporal modules on the two coarsest levels and the first two fusions."""

    def __init__(self, config: VideoDepthAnythingConfig, num_frames: int):
        super().__init__()
        out_channels, width = config.out_channels, config.features
        self.projects = nn.ModuleList(nn.Conv2d(config.embed_dim, channels, kernel_size=1) for channels in out_channels)
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = nn.Module()
        for index, channels in enumerate(out_channels, 1):
            setattr(
                self.scratch, f"layer{index}_rn", nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False)
            )
        for index in range(1, 5):
            setattr(self.scratch, f"refinenet{index}", FeatureFusionBlock(width))
        self.scratch.output_conv1 = nn.Conv2d(width, width // 2, kernel_size=3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(width // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(True),
            nn.Identity(),
        )
        self.motion_modules = nn.ModuleList(
            TemporalModule(channels, num_frames) for channels in (out_channels[2], out_channels[3], width, width)
        )

    def forward(
        self,
        features: list[tuple[torch.Tensor, torch.Tensor]],
        patch_h: int,
        patch_w: int,
        frames: int,
        micro_batch_size: int = 4,
    ) -> torch.Tensor:
        levels = []
        for index, (tokens, _cls_token) in enumerate(features):
            x = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w).contiguous()
            levels.append(self.resize_layers[index](self.projects[index](x)))
        layer_1, layer_2, layer_3, layer_4 = levels
        layer_3 = self.motion_modules[0](layer_3, frames)
        layer_4 = self.motion_modules[1](layer_4, frames)

        scratch = self.scratch
        layer_1_rn = scratch.layer1_rn(layer_1)
        layer_2_rn = scratch.layer2_rn(layer_2)
        layer_3_rn = scratch.layer3_rn(layer_3)
        layer_4_rn = scratch.layer4_rn(layer_4)
        path_4 = self.motion_modules[2](scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:]), frames)
        path_3 = self.motion_modules[3](scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:]), frames)

        # The two finest levels have no temporal mixing; decode them in micro-batches to bound memory.
        batch = layer_1_rn.shape[0]
        step = micro_batch_size if batch > micro_batch_size and batch % micro_batch_size == 0 else batch
        outputs = []
        for start in range(0, batch, step):
            window = slice(start, start + step)
            path_2 = scratch.refinenet2(path_3[window], layer_2_rn[window], size=layer_1_rn.shape[2:])
            path_1 = scratch.refinenet1(path_2, layer_1_rn[window])
            out = F.interpolate(
                scratch.output_conv1(path_1),
                (patch_h * PATCH_SIZE, patch_w * PATCH_SIZE),
                mode="bilinear",
                align_corners=True,
            )
            # The reference runs the last convolutions in FP32 even under autocast.
            with torch.autocast(device_type=out.device.type, enabled=False):
                outputs.append(scratch.output_conv2(out.float()).to(out.dtype))
        return torch.cat(outputs)


class VideoDepthAnything(nn.Module):
    def __init__(self, config: VideoDepthAnythingConfig, num_frames: int = 32):
        super().__init__()
        self.config = config
        self.pretrained = DinoVisionTransformer(config)
        self.head = DPTHeadTemporal(config, num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, 3, H, W] normalized frames (H, W multiples of 14) -> [B, T, H, W] depth."""
        batch, frames, _, height, width = x.shape
        features = self.pretrained.get_intermediate_layers(x.flatten(0, 1), self.config.intermediate_layers)
        depth = self.head(features, height // PATCH_SIZE, width // PATCH_SIZE, frames)
        depth = F.relu(F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=True))
        return depth.squeeze(1).unflatten(0, (batch, frames))
