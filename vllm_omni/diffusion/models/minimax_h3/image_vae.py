# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native MiniMax-H3-Image VAE: H3's causal CNN encoder and a 2D ViT decoder.

The image release stores one standard weight file under ``vae/``. Its encoder
and ``quant_conv`` are the H3 video VAE's encoder (same tensors); the decoder
is a 24-layer ViT over one latent frame with 2D RoPE. Every image runs as a
single frame, so this module keeps only the image paths of the released VAE:

- encode: ImageNet-normalized pixels, cropped top-left to the 16px grid, are
  encoded in 256px tiles blended over their 64px overlaps; the diagonal
  Gaussian posterior is *sampled* from the caller's generator;
- decode: latents are decoded in the same tiling with BF16 weights under BF16
  autocast (the released image recipe), blended, and mapped back to [0, 1].
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from vllm.utils.torch_utils import set_default_torch_dtype

MINIMAX_H3_IMAGE_VAE_RATIO = 16
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Fixed architecture of the released H3 visual VAE encoder.
_ENCODER_CH = 128
_ENCODER_CH_MULT = (1, 2, 2, 4, 4, 8)
_ENCODER_SPACE_DOWN = (2, 2, 2, 2, 1, 1)
_ENCODER_TIME_DOWN = (1, 2, 2, 1, 1, 1)
_ENCODER_RES_BLOCKS = 2


# ---------------------------------------------------------------------------
# Encoder: causal 3D CNN run on one frame
# ---------------------------------------------------------------------------


class _CausalConv3d(nn.Conv3d):
    """H3 VAE convolution: reflect spatial padding, causal zero temporal padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride=1, padding=0) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            padding_mode="reflect",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if sum(self.padding) == 0:
            return super().forward(x)
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0), mode="reflect")
        if x.shape[2] > 1:
            x = F.pad(x, (0, 0, 0, 0, self.padding[0] * 2, 0), mode="constant")
        else:
            x = F.pad(x, (0, 0, 0, 0, self.kernel_size[0] - 1, 0), mode="constant")
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


def _group_norm(channels: int) -> nn.GroupNorm:
    # The release isolates GroupNorm statistics per frame; one frame makes that
    # identical to a plain 5D GroupNorm.
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class _Downsample3d(nn.Module):
    def __init__(self, channels: int, *, time_stride: int, space_stride: int) -> None:
        super().__init__()
        self.space_stride = space_stride
        self.conv = _CausalConv3d(
            channels, channels, 3, stride=(time_stride, space_stride, space_stride), padding=(1, 0, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class _ResnetBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = _CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = _CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = _CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(F.silu(self.norm2(h), inplace=True))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return h.add_(x)


class _EncoderFCN3d(nn.Module):
    def __init__(self, z_channels: int) -> None:
        super().__init__()
        widths = [_ENCODER_CH * mult for mult in _ENCODER_CH_MULT]
        inputs = [widths[0], *widths[:-1]]
        self.conv_in = _CausalConv3d(3, widths[0], 3, padding=1)
        self.down = nn.ModuleList()
        for level, width in enumerate(widths):
            down = nn.Module()
            down.block = nn.ModuleList(
                [_ResnetBlock3d(inputs[level] if i == 0 else width, width) for i in range(_ENCODER_RES_BLOCKS)]
            )
            if _ENCODER_SPACE_DOWN[level] * _ENCODER_TIME_DOWN[level] > 1:
                down.downsample = _Downsample3d(
                    width, time_stride=_ENCODER_TIME_DOWN[level], space_stride=_ENCODER_SPACE_DOWN[level]
                )
            self.down.append(down)
        self.norm_out = _group_norm(widths[-1])
        self.conv_out = _CausalConv3d(widths[-1], 2 * z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h), inplace=True))


# ---------------------------------------------------------------------------
# Decoder: 2D ViT over one latent frame
# ---------------------------------------------------------------------------


def _token_ids_2d(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Length-normalized (H, W) token coordinates in (-1, 1), row major."""
    axes = []
    for size in (height, width):
        coords = torch.arange(0.5, size, dtype=dtype, device=device) / size
        axes.append(2.0 * coords - 1.0)
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 1).unsqueeze(0)


class _RotaryEmbedding2d(nn.Module):
    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError(f"2D RoPE dim {dim} must be divisible by 4")
        self.dim, self.theta = dim, theta
        self.register_buffer("inv_freq", self._fp32_inv_freq(), persistent=False)

    def _fp32_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        return 1 / self.theta ** torch.arange(0, 1, 4 / self.dim, dtype=torch.float32, device=device)

    def restore_fp32_inv_freq(self) -> None:
        """Recompute the bases in FP32 after a ``module.to(dtype)`` rounded them."""
        self.inv_freq = self._fp32_inv_freq(self.inv_freq.device)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(ids.device.type, enabled=False):
            angles = 2.0 * math.pi * ids[:, :, :, None] * self.inv_freq.to(ids.device)[None, None, None, :]
            angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
            cos, sin = torch.cos(angles), torch.sin(angles)
        return cos.to(ids.dtype), sin.to(ids.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.to(t.dtype), sin.to(t.dtype)
    rot = cos.shape[-1]
    t_rot, t_pass = t[..., :rot], t[..., rot:]
    rotated = t_rot * cos
    rotated.add_(_rotate_half(t_rot) * sin)
    return torch.cat((rotated, t_pass), dim=-1)


def _fp32_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # The released decoder runs its ViT norms in FP32 and casts back.
    return norm(x.float()).to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, heads: int, dim_head: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Per-head interleaved [q | k | v] layout.
        qkv = self.to_qkv(x).view(batch, seq, self.heads, 3 * self.dim_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = _apply_rope(_fp32_norm(self.norm_q, q), *rope)
        k = _apply_rope(_fp32_norm(self.norm_k, k), *rope)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.to_out(out.transpose(1, 2).reshape(batch, seq, -1))


class _FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner = dim * 4
        self.w1 = nn.Linear(dim, inner * 2)
        self.w2 = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate).mul_(up))


class _TransformerBlock(nn.Module):
    def __init__(self, heads: int, dim_head: int, eps: float) -> None:
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = _Attention(heads, dim_head, eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = _FeedForward(dim)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(_fp32_norm(self.norm1, x), rope) * self.scale1
        return x + self.ff(_fp32_norm(self.norm2, x)) * self.scale2


class _ViT2dDecoder(nn.Module):
    def __init__(self, config: dict[str, Any], *, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        expected = {
            "norm_type": "rms_norm",
            "qk_norm_type": "rms_norm",
            "qk_norm_affine": False,
            "norm_affine": True,
            "ffn_activation_fn": "silu",
            "ffn_use_gated": True,
        }
        for key, value in expected.items():
            if config.get(key, value) != value:
                raise ValueError(f"MiniMax H3 image VAE requires vit_decoder_kwargs.{key}={value!r}")
        heads, dim_head = int(config["heads"]), int(config["dim_head"])
        eps = float(config.get("eps", 1e-5))
        dim = heads * dim_head
        self.patch_size = MINIMAX_H3_IMAGE_VAE_RATIO
        self.pos_embed = _RotaryEmbedding2d(
            int(dim_head * float(config["rope_dim_ratio"])), float(config["rope_theta"])
        )
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, int(config.get("num_register_tokens", 4)), dim))
        self.transformer_blocks = nn.ModuleList(
            [_TransformerBlock(heads, dim_head, eps) for _ in range(int(config["num_layers"]))]
        )
        self.norm_out = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.proj_out = nn.Linear(dim, out_channels * self.patch_size**2)
        # Training-time masking token; part of the checkpoint, unused at inference.
        self.register_buffer("mask_token", torch.zeros(1, 1, dim))

    @staticmethod
    def _linear_no_autocast(linear: nn.Linear, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=False):
            return linear(x.to(linear.weight.dtype)).to(out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, h, w]`` latents to ``[B, 3, 16h, 16w]`` normalized pixels."""
        batch, _, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        hidden = self._linear_no_autocast(self.x_embedder, tokens, tokens.dtype)
        num_patches = hidden.shape[1]
        suffix = [self.register_tokens.expand(batch, -1, -1), torch.zeros_like(hidden[:, :1])]
        hidden = torch.cat([hidden, *suffix], dim=1)
        ids = _token_ids_2d(height, width, x.device, x.dtype)
        ids = torch.cat([ids, ids.new_zeros(1, hidden.shape[1] - num_patches, 2)], dim=1)
        rope_dtype = (
            torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else hidden.dtype
        )
        cos, sin = self.pos_embed(ids)
        rope = (cos.to(rope_dtype), sin.to(rope_dtype))
        for block in self.transformer_blocks:
            hidden = block(hidden, rope)
        hidden = self.norm_out(hidden)
        out = self._linear_no_autocast(self.proj_out, hidden, hidden.dtype)[:, :num_patches]
        patch = self.patch_size
        out = out.view(batch, height, width, -1, patch, patch).permute(0, 3, 1, 4, 2, 5)
        return out.reshape(batch, -1, height * patch, width * patch)


# ---------------------------------------------------------------------------
# Tiling (released VAE: 256px tiles, 64px minimum overlap, linear blending)
# ---------------------------------------------------------------------------


def _split_tiles(length: int, tile: int, overlap_min: int) -> tuple[list[int], list[int], list[int]]:
    if tile >= length:
        return [0], [length], []
    count = math.ceil(length / tile)
    while True:
        overlaps = [overlap_min] * (count - 1)
        remaining = tile * count - sum(overlaps) - length
        if remaining >= 0:
            break
        count += 1
    for index in range(remaining // MINIMAX_H3_IMAGE_VAE_RATIO):
        overlaps[index % (count - 1)] += MINIMAX_H3_IMAGE_VAE_RATIO
    starts = [0]
    for index in range(count - 1):
        starts.append(starts[-1] + tile - overlaps[index])
    return starts, [tile] * count, overlaps


def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a = (1 - positions / extent).view(shape)
    weight_b = (positions / extent).view(shape)
    blended = a.narrow(dim, a.shape[dim] - extent, extent) * weight_a
    blended.add_(b.narrow(dim, 0, extent) * weight_b)
    if extent < b.shape[dim]:
        return torch.cat([blended, b.narrow(dim, extent, b.shape[dim] - extent)], dim=dim)
    return blended


def _assemble_tiles(rows: list[list[torch.Tensor]], y_overlap: Sequence[int], x_overlap: Sequence[int]) -> torch.Tensor:
    last_row, last_col = len(rows) - 1, len(rows[0]) - 1
    height = sum(row[0].shape[-2] - (y_overlap[i] if i < last_row else 0) for i, row in enumerate(rows))
    width = sum(tile.shape[-1] - (x_overlap[j] if j < last_col else 0) for j, tile in enumerate(rows[0]))
    output = rows[0][0].new_empty((*rows[0][0].shape[:-2], height, width))
    y_offset = 0
    # The release blends vertically before horizontally.
    for i, row in enumerate(rows):
        row_height = row[0].shape[-2] - (y_overlap[i] if i < last_row else 0)
        x_offset = 0
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = _blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
            if i < last_row:
                tile = tile[..., : -y_overlap[i], :]
            if j < last_col:
                tile = tile[..., :, : -x_overlap[j]]
            output[..., y_offset : y_offset + tile.shape[-2], x_offset : x_offset + tile.shape[-1]].copy_(tile)
            x_offset += tile.shape[-1]
        y_offset += row_height
    return output


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class MiniMaxH3ImageVAE(nn.Module):
    """Native image VAE of MiniMax-H3-Image (``vae/`` of the release)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        if config.get("_class_name") != "MiniMaxH3ImageVAE" or not config.get("vit_decoder_2d"):
            raise ValueError("expected a MiniMaxH3ImageVAE config with vit_decoder_2d=true")
        channels = int(config["latent_channels"])
        for key in ("latents_mean", "latents_std"):
            if len(config[key]) != channels:
                raise ValueError(f"vae {key} must contain {channels} values")
        if int(config.get("vae_token_drop", 0)) != 0:
            raise ValueError("MiniMax H3 image VAE requires vae_token_drop=0")
        self.latent_channels = channels
        self.encoder_tiling = bool(config.get("vae_encoder_tiling", 1))
        self.decoder_tiling = bool(config.get("vae_decoder_tiling", 1))
        self.tile_size = int(config.get("vae_tile_size", 256))
        self.tile_overlap_min = int(config.get("vae_tile_overlap_min", 64))
        self.encoder = _EncoderFCN3d(channels)
        self.quant_conv = nn.Conv3d(2 * channels, 2 * channels, 1)
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)
        self._decode_graphs: _TileGraphs | None = None
        self.decoder = _ViT2dDecoder(config["vit_decoder_kwargs"], in_channels=channels)
        fp32 = torch.float32
        self.register_buffer(
            "latents_mean", torch.tensor(config["latents_mean"], dtype=fp32).view(1, -1, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(config["latents_std"], dtype=fp32).view(1, -1, 1, 1, 1), persistent=False
        )
        self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN, dtype=fp32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD, dtype=fp32).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> MiniMaxH3ImageVAE:
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        # Pipelines are built under the DiT's default dtype (BF16); the released
        # VAE weights and statistics are FP32 and must not be rounded on load.
        with set_default_torch_dtype(torch.float32), torch.device("cpu"):
            model = cls(config)
        state = load_file(str(path / "diffusion_pytorch_model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"MiniMax H3 image VAE weights mismatch: missing={missing}, unexpected={unexpected}")
        return model.eval().float()

    @property
    def device(self) -> torch.device:
        return self.quant_conv.weight.device

    # -- encode -------------------------------------------------------------

    def _tiled(self, x: torch.Tensor, fn, *, latent_in: bool, group: Any = None) -> torch.Tensor:
        """Run ``fn`` over the released tile grid and blend the results.

        With a process group of several ranks, tiles are assigned round-robin
        and gathered: each tile still runs alone with the same kernels, so the
        blended output is identical to the single-rank result.
        """
        ratio = MINIMAX_H3_IMAGE_VAE_RATIO
        scale = ratio if latent_in else 1
        height, width = x.shape[-2] * scale, x.shape[-1] * scale
        y_idx, y_len, y_overlap = _split_tiles(height, self.tile_size, self.tile_overlap_min)
        x_idx, x_len, x_overlap = _split_tiles(width, self.tile_size, self.tile_overlap_min)
        tiles = [
            x[..., top // scale : (top + tile_h) // scale, left // scale : (left + tile_w) // scale]
            for top, tile_h in zip(y_idx, y_len)
            for left, tile_w in zip(x_idx, x_len)
        ]
        outputs = _run_tiles(tiles, fn, group)
        columns = len(x_idx)
        rows = [outputs[row * columns : (row + 1) * columns] for row in range(len(y_idx))]
        if latent_in:
            return _assemble_tiles(rows, y_overlap, x_overlap)
        return _assemble_tiles(rows, [o // ratio for o in y_overlap], [o // ratio for o in x_overlap])

    def _encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))

    @torch.inference_mode()
    def encode_image_latent(self, image: Any, *, generator: torch.Generator, group: Any = None) -> torch.Tensor:
        """Sample the raw latent ``[1, C, 1, h, w]`` (FP16-rounded, FP32) of an RGB image.

        The image is cropped top-left to the 16px grid. The posterior noise is
        drawn on the generator's device, which makes the draw part of the
        request's RNG stream.
        """
        import numpy as np

        array = np.array(image.convert("RGB") if hasattr(image, "convert") else image, dtype=np.uint8)
        pixels = torch.from_numpy(array).to(self.device)[None].permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
        pixels = (pixels - self.pixel_mean) / self.pixel_std
        ratio = MINIMAX_H3_IMAGE_VAE_RATIO
        height, width = pixels.shape[-2] // ratio * ratio, pixels.shape[-1] // ratio * ratio
        if height == 0 or width == 0:
            raise ValueError("image must be at least 16 pixels on each axis")
        pixels = pixels[:, :, :height, :width].unsqueeze(2)
        if self.encoder_tiling:
            moments = self._tiled(pixels, self._encode_moments, latent_in=False, group=group)
        else:
            moments = self._encode_moments(pixels)
        moments = moments.to(torch.float32)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = logvar.clamp(-30.0, 20.0).mul(0.5).exp_()
        noise = torch.randn(mean.shape, generator=generator, device=generator.device).to(mean.device)
        latent = noise.mul_(std).add_(mean)
        # The release keeps image latents in FP16.
        return latent.to(torch.float16).to(torch.float32)

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latents_mean.to(latent.device)) / self.latents_std.to(latent.device)

    # -- decode -------------------------------------------------------------

    def _decode_tile(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z.squeeze(2))).unsqueeze(2)

    def prepare_decoder_weights(self, dtype: torch.dtype = torch.bfloat16) -> None:
        """Cast the decoder weights to ``dtype``; the RoPE bases stay FP32 because
        the angle math runs in FP32 and rounded bases shift every frequency."""
        for module in (self.post_quant_conv, self.decoder):
            ropes = [m for m in module.modules() if isinstance(m, _RotaryEmbedding2d)]
            rope_buffers = {id(m.inv_freq) for m in ropes}
            stale = any(
                t.is_floating_point() and t.dtype != dtype
                for t in (*module.parameters(), *module.buffers())
                if id(t) not in rope_buffers
            ) or any(m.inv_freq.dtype != torch.float32 for m in ropes)
            if stale:
                module.to(dtype)
                for m in ropes:
                    m.restore_fp32_inv_freq()
                self._decode_graphs = None

    @torch.inference_mode()
    def decode_latent(
        self,
        latent: torch.Tensor,
        *,
        group: Any = None,
        use_cuda_graph: bool = False,
        tile_batch_size: int = 1,
    ) -> torch.Tensor:
        """Normalized ``[B, C, 1, h, w]`` latents to ``[B, 3, 16h, 16w]`` pixels in [0, 1].

        ``group`` spreads the tiles over its ranks (every rank returns the full
        image, decoded from the first rank's latent). ``use_cuda_graph`` replays
        one captured graph per tile shape; the kernels are those of the eager
        single-tile decode, so the output is unchanged.

        ``tile_batch_size > 1`` opts into batched tile kernels. Tile boundaries
        and blending are unchanged, but floating-point results may differ.
        """
        if type(tile_batch_size) is not int or not 1 <= tile_batch_size <= 32:
            raise ValueError("tile_batch_size must be an integer between 1 and 32")
        if tile_batch_size != 1 and (not use_cuda_graph or self.device.type != "cuda"):
            raise ValueError("batched tile decode requires CUDA graphs")
        raw = latent.to(device=self.device, dtype=torch.float32)
        raw = raw * self.latents_std.to(raw.device) + self.latents_mean.to(raw.device)
        if _group_world(group)[0] > 1:
            raw = raw.contiguous()
            dist.broadcast(raw, src=dist.get_global_rank(group, 0), group=group)
        self.prepare_decoder_weights(torch.bfloat16)
        decode_tile = self._decode_tile
        if use_cuda_graph and raw.device.type == "cuda":
            if self._decode_graphs is None or self._decode_graphs.tile_batch_size != tile_batch_size:
                self._decode_graphs = (
                    _TileGraphs(self._decode_tile, autocast_dtype=torch.bfloat16)
                    if tile_batch_size == 1
                    else _BatchedTileGraphs(
                        self._decode_tile, autocast_dtype=torch.bfloat16, tile_batch_size=tile_batch_size
                    )
                )
            decode_tile = self._decode_graphs
        with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16):
            if self.decoder_tiling:
                decoded = self._tiled(raw, decode_tile, latent_in=True, group=group)
            else:
                decoded = decode_tile(raw)
            decoded = decoded.squeeze(2)
            pixels = decoded * self.pixel_std.to(decoded.dtype) + self.pixel_mean.to(decoded.dtype)
        return pixels.float().clamp_(0, 1)


def _group_world(group: Any) -> tuple[int, int]:
    if group is None or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


def _run_tiles(tiles: list[torch.Tensor], fn, group: Any) -> list[torch.Tensor]:
    """``[fn(tile) for tile in tiles]``, with tiles assigned round-robin over ``group``."""
    world, rank = _group_world(group)
    if world == 1 or len(tiles) < world:
        # Fewer tiles than ranks: every rank runs the whole grid locally.
        return fn.map(tiles) if hasattr(fn, "map") else [fn(tile) for tile in tiles]
    mine = [tiles[index] for index in range(rank, len(tiles), world)]
    local = fn.map(mine) if hasattr(fn, "map") else [fn(tile) for tile in mine]
    shape, dtype = local[0].shape, local[0].dtype
    if any(output.shape != shape or output.dtype != dtype for output in local):
        raise ValueError("MiniMax H3 image VAE tiles must share one output shape")
    per_rank = -(-len(tiles) // world)
    stacked = local[0].new_zeros((per_rank, *shape))
    torch.stack(local, out=stacked[: len(local)])
    gathered = stacked.new_empty((world * per_rank, *shape))
    dist.all_gather_into_tensor(gathered, stacked, group=group)
    return [gathered[(index % world) * per_rank + index // world] for index in range(len(tiles))]


class _TileGraphs:
    """CUDA graphs of ``fn`` per tile input shape, replayed on parallel streams.

    Each graph replays ``fn``'s eager single-tile kernels. One tile leaves most
    of the GPU idle, so ``map`` spreads tiles round-robin over ``replicas``
    independent graphs (own buffers and stream); every tile still runs the
    same kernels, so the outputs are unchanged.
    """

    max_shapes = 4
    # Concurrent replica replay can hang on ROCm (gfx950); one replica does not.
    replicas = 1 if torch.version.hip else 6
    tile_batch_size = 1

    def __init__(self, fn, *, autocast_dtype: torch.dtype) -> None:
        self.fn = fn
        self.autocast_dtype = autocast_dtype
        self.entries: dict[tuple, list[tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.cuda.Stream]]] = {}

    def _capture(self, z: torch.Tensor):
        static_in = z.clone()
        stream = torch.cuda.Stream(device=z.device)
        stream.wait_stream(torch.cuda.current_stream(z.device))
        with torch.cuda.stream(stream), torch.autocast(z.device.type, dtype=self.autocast_dtype):
            # Warm up outside the graph (lazy allocations, kernel selection).
            for _ in range(2):
                self.fn(static_in)
        torch.cuda.current_stream(z.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # Capture on the replica's own stream. torch.cuda.graph() otherwise
        # captures every replica on one shared side stream, so all replicas bake
        # in that stream's cuBLAS workspace and race on it when replayed
        # concurrently in map() (nondeterministic pixels; NCCL hangs observed on
        # 4x RTX PRO 6000).
        with torch.cuda.graph(graph, stream=stream), torch.autocast(z.device.type, dtype=self.autocast_dtype):
            static_out = self.fn(static_in)
        return graph, static_in, static_out, stream

    def _replicas(self, z: torch.Tensor, count: int):
        key = (tuple(z.shape), z.dtype, z.device)
        entries = self.entries.get(key)
        if entries is None:
            if len(self.entries) >= self.max_shapes:
                self.entries.pop(next(iter(self.entries)))
            entries = self.entries[key] = []
        while len(entries) < count:
            entries.append(self._capture(z))
        return entries

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.map([z])[0]

    def map(self, tiles: list[torch.Tensor]) -> list[torch.Tensor]:
        if not tiles:
            return []
        if any(tile.shape != tiles[0].shape for tile in tiles):
            return [self.map([tile])[0] for tile in tiles]
        entries = self._replicas(tiles[0], min(self.replicas, len(tiles)))
        current = torch.cuda.current_stream(tiles[0].device)
        for *_, stream in entries:
            stream.wait_stream(current)
        outputs = []
        for index, tile in enumerate(tiles):
            graph, static_in, static_out, stream = entries[index % len(entries)]
            with torch.cuda.stream(stream):
                static_in.copy_(tile)
                graph.replay()
                output = static_out.clone()
            # The consumer stream reads the output after the replica's stream.
            output.record_stream(current)
            outputs.append(output)
        for *_, stream in entries:
            current.wait_stream(stream)
        return outputs


class _BatchedTileGraphs(_TileGraphs):
    """Opt-in tile batching with independent full-batch and tail graphs."""

    def __init__(self, fn, *, autocast_dtype: torch.dtype, tile_batch_size: int) -> None:
        super().__init__(fn, autocast_dtype=autocast_dtype)
        self.tile_batch_size = tile_batch_size

    def map(self, tiles: list[torch.Tensor]) -> list[torch.Tensor]:
        if not tiles or any(tile.shape != tiles[0].shape for tile in tiles):
            return super().map(tiles)
        batches = [
            torch.cat(tiles[start : start + self.tile_batch_size], dim=0)
            for start in range(0, len(tiles), self.tile_batch_size)
        ]
        full = len(tiles) // self.tile_batch_size
        outputs = super().map(batches[:full]) + super().map(batches[full:])
        return [tile for batch in outputs for tile in batch.split(tiles[0].shape[0])]


__all__ = ["MINIMAX_H3_IMAGE_VAE_RATIO", "MiniMaxH3ImageVAE"]
