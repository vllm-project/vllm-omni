# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Adapted from diffusers `AutoencoderKL` (v0.40.0):
# diffusers/models/autoencoders/{autoencoder_kl.py, vae.py},
# diffusers/models/unets/unet_2d_blocks.py, upsampling.py, downsampling.py.

"""Boogu-private AutoencoderKL with GroupNorm+SiLU fusion in the owned forward.

Why this exists (#6686, decision D-15): GroupNorm followed by SiLU is ~55% of
``vae.decode`` on Boogu-Image — 29 sites, all in the decoder. Fusing them by
patching diffusers' modules requires trusting that diffusers' block forwards
call the activation immediately after the norm, which an attribute-level guard
cannot verify. Owning the forward removes that trust: this module is a faithful
copy of diffusers' ``AutoencoderKL`` for Boogu's FLUX-style configuration, with
:func:`fused_group_norm_silu` called exactly where this file's own forwards put
it. Precedent: HunyuanImage3's in-repo ``AutoencoderKLConv3D`` (#6306).

Fidelity contract:

- **Owned dataflow, reused leaves.** Only the block forwards are ours; the
  parameterised leaves are stock ``nn.GroupNorm`` / ``nn.Conv2d`` and the mid
  block runs diffusers' own ``Attention``, so numerics reduce to the fused
  op's ~1-ulp envelope (unfused paths are bit-identical to diffusers).
- **Submodule names mirror diffusers exactly** (``decoder.up_blocks[i]
  .resnets[j]``, ``mid_block.attentions[0]``, ``conv_norm_out``, ...), so the
  stock Boogu ``vae/`` checkpoint loads with zero key remapping.
- The fusion sits behind a runtime switch (:meth:`BooguAutoencoderKL
  .set_group_norm_silu_fusion`) and is only ever armed when
  ``config.act_fn == "silu"``; with the switch off every site computes the
  plain ``act(norm(x))``, and the fused op itself falls back natively where
  Triton is unavailable.
- Tiling and slicing are not implemented (the Boogu pipeline never enables
  them); the enable methods raise rather than silently diverge.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.ops.fused_group_norm_silu import (
    fused_group_norm_silu,
)

logger = init_logger(__name__)

# Kill-switch, identical keys to the T5/T8 install path: additional_config
# (bool, default on) with a disable-only env override.
CONFIG_KEY = "vae_group_norm_silu_fusion"
ENV_KEY = "VLLM_OMNI_VAE_GN_SILU_FUSION"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def vae_group_norm_silu_fusion_enabled(od_config: Any) -> bool:
    """Resolve the kill-switch: on by default; ``additional_config
    {"vae_group_norm_silu_fusion": false}`` or ``VLLM_OMNI_VAE_GN_SILU_FUSION=0``
    disables it."""
    if not _as_bool(os.environ.get(ENV_KEY), True):
        return False
    additional_config = getattr(od_config, "additional_config", None) or {}
    return _as_bool(additional_config.get(CONFIG_KEY), True)


class BooguVaeStrictLoadError(Exception):
    """The in-repo VAE could not load a checkpoint with a complete, exact key
    match — a bug in THIS module, to be caught by the pipeline's diffusers
    fallback, not papered over. Deliberately not an OSError/RuntimeError/
    ValueError so from_pretrained_with_prefetch's cache-heal retry does not
    swallow it."""


def _norm_silu(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    return fused_group_norm_silu(x, norm.weight, norm.bias, norm.num_groups, norm.eps)


class BooguResnetBlock2D(nn.Module):
    """diffusers ``ResnetBlock2D`` for the temb-free VAE case, forward owned.

    Parameter names (norm1/conv1/norm2/conv2/conv_shortcut) mirror diffusers.
    ``fuse_group_norm_silu`` selects, per call, the fused GroupNorm+SiLU kernel
    or the plain ``act(norm(x))`` pair; it is armed only by the parent module
    and only when the activation is SiLU.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        groups: int,
        eps: float,
        act_fn: str,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.fuse_group_norm_silu = False

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = get_activation(act_fn)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        # The whole point of T9: the norm -> activation adjacency is a fact of
        # THIS forward, not an assumption about a third-party one.
        hidden_states = input_tensor
        if self.fuse_group_norm_silu:
            hidden_states = _norm_silu(self.norm1, hidden_states)
        else:
            hidden_states = self.nonlinearity(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        if self.fuse_group_norm_silu:
            hidden_states = _norm_silu(self.norm2, hidden_states)
        else:
            hidden_states = self.nonlinearity(self.norm2(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor


class BooguUpsample2D(nn.Module):
    """diffusers ``Upsample2D(use_conv=True)``: nearest 2x then a 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # diffusers works around two upsample_nearest issues; replicate both
        # (they only force contiguity, never values).
        if hidden_states.shape[0] >= 64 or hidden_states.numel() * 2 > 2**31:
            hidden_states = hidden_states.contiguous()
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class BooguDownsample2D(nn.Module):
    """diffusers ``Downsample2D(use_conv=True, padding=0, name="op")``:
    asymmetric (0,1,0,1) pad then a stride-2 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class BooguUpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                BooguResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    groups=groups,
                    eps=eps,
                    act_fn=act_fn,
                )
                for i in range(num_layers)
            ]
        )
        self.upsamplers = nn.ModuleList([BooguUpsample2D(out_channels)]) if add_upsample else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class BooguDownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_downsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                BooguResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    groups=groups,
                    eps=eps,
                    act_fn=act_fn,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = nn.ModuleList([BooguDownsample2D(out_channels)]) if add_downsample else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class BooguMidBlock2D(nn.Module):
    """diffusers ``UNetMidBlock2D`` for the VAE case: resnet, attention, resnet.

    The attention IS diffusers' ``Attention`` (deprecated-attn-block flavour,
    exactly as ``UNetMidBlock2D`` constructs it), so its numerics and its
    checkpoint keys (``attentions.0.group_norm/to_q/to_k/to_v/to_out.0``) are
    inherited rather than re-implemented.
    """

    def __init__(self, *, in_channels: int, groups: int, eps: float, act_fn: str, add_attention: bool) -> None:
        super().__init__()

        def _resnet() -> BooguResnetBlock2D:
            return BooguResnetBlock2D(
                in_channels=in_channels, out_channels=in_channels, groups=groups, eps=eps, act_fn=act_fn
            )

        self.resnets = nn.ModuleList([_resnet(), _resnet()])
        attention = None
        if add_attention:
            attention = Attention(
                in_channels,
                heads=1,
                dim_head=in_channels,
                rescale_output_factor=1.0,
                eps=eps,
                norm_num_groups=groups,
                spatial_norm_dim=None,
                residual_connection=True,
                bias=True,
                upcast_softmax=True,
                _from_deprecated_attn_block=True,
            )
        self.attentions = nn.ModuleList([attention])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        if self.attentions[0] is not None:
            hidden_states = self.attentions[0](hidden_states, temb=None)
        return self.resnets[1](hidden_states)


class BooguDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        norm_num_groups: int,
        act_fn: str,
        mid_block_add_attention: bool,
    ) -> None:
        super().__init__()
        self.fuse_conv_norm_out = False

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = BooguMidBlock2D(
            in_channels=block_out_channels[-1],
            groups=norm_num_groups,
            eps=1e-6,
            act_fn=act_fn,
            add_attention=mid_block_add_attention,
        )

        self.up_blocks = nn.ModuleList([])
        reversed_channels = list(reversed(block_out_channels))
        output_channel = reversed_channels[0]
        for i, ch in enumerate(reversed_channels):
            prev_output_channel = output_channel
            output_channel = ch
            self.up_blocks.append(
                BooguUpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    groups=norm_num_groups,
                    eps=1e-6,
                    act_fn=act_fn,
                    add_upsample=i != len(block_out_channels) - 1,
                )
            )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        # diffusers hardcodes SiLU here regardless of act_fn — mirror that.
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        if self.fuse_conv_norm_out:
            sample = _norm_silu(self.conv_norm_out, sample)
        else:
            sample = self.conv_act(self.conv_norm_out(sample))
        return self.conv_out(sample)


class BooguEncoder(nn.Module):
    """Owned but plain: same composed leaves, no fused op (encoder fusion is
    out of scope for v1 so the perf claim stays aligned with T4's
    decoder-only measurement)."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        norm_num_groups: int,
        act_fn: str,
        mid_block_add_attention: bool,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = ch
            self.down_blocks.append(
                BooguDownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    groups=norm_num_groups,
                    eps=1e-6,
                    act_fn=act_fn,
                    add_downsample=i != len(block_out_channels) - 1,
                )
            )

        self.mid_block = BooguMidBlock2D(
            in_channels=block_out_channels[-1],
            groups=norm_num_groups,
            eps=1e-6,
            act_fn=act_fn,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        # diffusers hardcodes SiLU here regardless of act_fn — mirror that.
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        sample = self.conv_act(self.conv_norm_out(sample))
        return self.conv_out(sample)


class BooguAutoencoderKL(ModelMixin, ConfigMixin):
    """Boogu-private AutoencoderKL; drop-in for diffusers' at the pipeline
    boundary (``from_pretrained`` / ``.config`` / ``encode`` / ``decode``)."""

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,
        latents_mean: tuple[float] | None = None,
        latents_std: tuple[float] | None = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
        if any(t != "DownEncoderBlock2D" for t in down_block_types) or any(
            t != "UpDecoderBlock2D" for t in up_block_types
        ):
            raise NotImplementedError(
                "BooguAutoencoderKL implements the plain DownEncoderBlock2D/UpDecoderBlock2D "
                f"architecture only; got {down_block_types} / {up_block_types}"
            )

        self.encoder = BooguEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.decoder = BooguDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

        # Fusion is only ever armed for the activation the fused op implements.
        self._fusion_supported = act_fn == "silu"
        self.set_group_norm_silu_fusion(self._fusion_supported)

    # -- strict loading ------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """diffusers' loader only *warns* on missing/unexpected/mismatched
        keys; this module must match the checkpoint exactly or refuse it, so
        the pipeline can fall back to diffusers rather than serve a
        partially-initialised VAE."""
        kwargs["output_loading_info"] = True
        model, info = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        problems = {k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info.get(k)}
        if problems:
            raise BooguVaeStrictLoadError(
                f"strict load failed for {pretrained_model_name_or_path}: {problems}"
            )
        return model

    # -- fusion control ----------------------------------------------------

    def _decoder_fusion_sites(self):
        for up_block in self.decoder.up_blocks:
            yield from up_block.resnets
        yield from self.decoder.mid_block.resnets

    def set_group_norm_silu_fusion(self, enabled: bool) -> int:
        """Arm or disarm the fused GroupNorm+SiLU sites in the decoder.

        Returns the number of fused sites now active (0 when disabled or when
        the configured activation is not SiLU — the fusion never applies to a
        non-SiLU activation).
        """
        enabled = bool(enabled) and self._fusion_supported
        for resnet in self._decoder_fusion_sites():
            resnet.fuse_group_norm_silu = enabled
        self.decoder.fuse_conv_norm_out = enabled
        return self.group_norm_silu_fusion_site_count()

    def group_norm_silu_fusion_site_count(self) -> int:
        count = sum(2 for r in self._decoder_fusion_sites() if r.fuse_group_norm_silu)
        return count + (1 if self.decoder.fuse_conv_norm_out else 0)

    # -- tiling / slicing: not implemented, refuse loudly -------------------

    def enable_tiling(self, *args, **kwargs) -> None:
        raise NotImplementedError("BooguAutoencoderKL does not implement tiling")

    def enable_slicing(self) -> None:
        raise NotImplementedError("BooguAutoencoderKL does not implement slicing")

    # -- encode / decode (diffusers' contract, minus tiling/slicing) --------

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None) -> DecoderOutput | torch.Tensor:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        decoded = self.decoder(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | torch.Tensor:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
