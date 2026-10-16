# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flux3 Image VAE: a packed image autoencoder (loads ae.safetensors).

Self-contained implementation. The module layout (submodule names, buffers) is
fixed so ``ae.safetensors`` loads with ``strict=True``.

Architecture (defaults, ``AutoEncoderParams``):
  ch=128, ch_mult=[1,2,4,4] → 8x spatial downsample in the conv encoder,
  z_channels=32, pixel-shuffle packing ps=[2,2] → total 16x spatial downsample
  and 32*4=128 latent channels, followed by an affine-free BatchNorm2d.

Tensor contracts (AutoEncoder.encode / .decode):
- encode: (B, 3, H, W) in [-1, 1] → (B, 128, H//16, W//16)  (BN-normalised)
- decode: (B, 128, h, w)          → (B, 3, 16*h, 16*w)
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from einops import rearrange
from torch import Tensor, nn

from vllm_omni.model_executor.models.common.ops import fused_group_norm_silu

# ---------------------------------------------------------------------------
# Architecture. Keep module/attribute names identical so ae.safetensors loads
# with strict=True.
# ---------------------------------------------------------------------------


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def group_norm_silu(x: Tensor, norm: nn.GroupNorm) -> Tensor:
    """Use the fused Triton path on CUDA and the native path elsewhere."""
    if not x.is_cuda:
        return swish(norm(x))
    return fused_group_norm_silu(
        x,
        norm.weight,
        norm.bias,
        num_groups=norm.num_groups,
        eps=norm.eps,
        autotune=False,
    )


class _UnfusedAttn(nn.Module):
    """Unfused single-head attention using F.scaled_dot_product_attention."""

    def forward(self, q: Tensor, k: Tensor, v: Tensor, b: int, c: int, h: int, w: int) -> Tensor:
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.attn: nn.Module = _UnfusedAttn()

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        return self.attn(q, k, v, b, c, h, w)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = group_norm_silu(h, self.norm1)
        h = self.conv1(h)

        h = group_norm_silu(h, self.norm2)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = group_norm_silu(h, self.norm_out)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams = AutoEncoderParams()):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def pack(self, mean: Tensor) -> Tensor:
        """(B, C, H, W) → (B, C*ps0*ps1, H//ps0, W//ps1) pixel-unshuffle."""
        return rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )

    def unpack(self, z: Tensor) -> Tensor:
        """(B, C*ps0*ps1, h, w) → (B, C, h*ps0, w*ps1) pixel-shuffle."""
        return rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = self.pack(mean)
        z = self.normalize(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = self.unpack(z)
        dec = self.decoder(z)
        return dec


# ---------------------------------------------------------------------------
# Flux3ImageVAE — vllm-omni integration wrapper (nn.Module)
# ---------------------------------------------------------------------------


class Flux3ImageVAE(nn.Module):
    """vllm-omni wrapper around the packed image AutoEncoder.

    Constructed from configuration and populated by the pipeline's unified
    component loader.

    Normalization (affine-free BatchNorm) and pixel-shuffle packing are internal
    to ``encode``/``decode`` — callers must NOT apply an external BN correction or
    patchify step.
    """

    def __init__(
        self,
        device: str = "cuda",
        params: AutoEncoderParams | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if params is None:
            params = AutoEncoderParams()

        inner = AutoEncoder(params)
        inner.to(device)
        if dtype is not None:
            inner.to(dtype)
        inner.eval()

        self._model = inner
        self._device = device

        # ``block_out_channels`` / ``latent_channels`` mirror the config keys the
        # pipeline reads off diffusers VAEs, so downstream scale-factor math
        # (image_vae_scale_factor = 2**(len(block_out_channels)-1)) is unchanged.
        self.config = SimpleNamespace(
            block_out_channels=[params.ch * m for m in params.ch_mult],
            latent_channels=params.z_channels,
            batch_norm_eps=inner.bn_eps,
            ps=list(inner.ps),
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self._model.parameters()).dtype

    @property
    def bn(self) -> nn.Module:
        return self._model.bn

    @torch.inference_mode()
    def encode(self, image: Tensor) -> Tensor:
        """Encode an image to BN-normalised packed latents.

        Args:
            image: (B, 3, H, W) in [-1, 1]; H, W must be divisible by 16.

        Returns:
            Normalised latents (B, z_channels*4, H//16, W//16).
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4 (B, 3, H, W), got {image.ndim}.")
        if image.dtype != self.dtype:
            image = image.to(self.dtype)
        return self._model.encode(image)

    @torch.inference_mode()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode BN-normalised packed latents back to an image.

        Args:
            latents: (B, z_channels*4, h, w)

        Returns:
            Image (B, 3, 16*h, 16*w).
        """
        if latents.dtype != self.dtype:
            latents = latents.to(self.dtype)
        return self._model.decode(latents)

    @torch.inference_mode()
    def to_pixel_latents(self, latents: Tensor) -> Tensor:
        """Denormalise + unpack packed latents to per-pixel latents (no decoder).

        Implements the ``output_type == "latent"`` path: the
        BN-normalised packed latents (B, z_channels*4, h, w) are inverse-normalised
        and pixel-shuffled back to (B, z_channels, 2*h, 2*w) without running the
        conv decoder.
        """
        z = self._model.inv_normalize(latents)
        return self._model.unpack(z)
