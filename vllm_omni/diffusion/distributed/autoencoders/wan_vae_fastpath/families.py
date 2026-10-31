# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan-family VAE implementations the fast path binds to.

Several VAEs are Wan's causal VAE under other class names: diffusers' ``AutoencoderKLQwenImage`` (Qwen-Image,
Qwen-Image-Edit, Anima, Krea2, ...), vLLM-Omni's copy of it (Qwen-Image-Layered) and vLLM-Omni's
``AutoencoderKLQwenImage21`` (Qwen-Image-2.1, Wan 2.2's residual VAE run one image at a time). Their module forwards
are the Wan ones up to names, so one set of replacement forwards serves all of them. A :class:`WanVaeFamily` names the
class that plays each role in one implementation, plus the few places where implementations differ:

* ``causal_conv_kind``: ``"conv3d"`` (``nn.Conv3d`` with a temporal feature cache, Wan) or ``"conv2d_single_frame"``
  (``nn.Conv2d`` on the squeezed frame, never cached, Qwen-Image-2.1).
* ``up_blocks_take_first_chunk``: whether the decoder passes ``first_chunk`` to its up blocks.

A new Wan-style VAE gets the fast path by registering a family (:func:`register_family`) rather than by copying the
forwards. Module types are matched exactly (``type(m) is cls``), so subclasses with their own forwards (for example
the spatially sharded convolutions) are never bound by accident.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

from diffusers.models.autoencoders import autoencoder_kl_wan as wan
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

CAUSAL_CONV_KINDS = ("conv3d", "conv2d_single_frame")


@dataclass(frozen=True)
class WanVaeFamily:
    """The classes that play each fast-path role in one Wan-style VAE implementation."""

    name: str
    vaes: tuple[type[nn.Module], ...]
    decoder: type[nn.Module]
    encoder: type[nn.Module] | None
    causal_conv: type[nn.Module]
    # ``None`` for Wan: ``forwards.is_diffusers_rms_norm`` identifies ``WanRMS_norm`` by name, because the Wan NPU patch
    # rebinds that name to ``RMSNormVAE`` (different numerics) in diffusers' namespace.
    rms_norm: type[nn.Module] | None
    resample: type[nn.Module]
    upsample: type[nn.Module]
    residual_block: type[nn.Module]
    residual_up_block: type[nn.Module] | None = None
    dup_up: type[nn.Module] | None = None
    attention_block: type[nn.Module] | None = None
    causal_conv_kind: str = "conv3d"
    up_blocks_take_first_chunk: bool = True
    # channels_last level only: single-head attention as chunked low-precision matmuls (see forwards).
    chunked_attention: bool = False

    def __post_init__(self) -> None:
        if self.causal_conv_kind not in CAUSAL_CONV_KINDS:
            raise ValueError(f"causal_conv_kind must be one of {CAUSAL_CONV_KINDS}, got {self.causal_conv_kind!r}")

    def decoder_roles(self) -> dict[type[nn.Module], str]:
        roles = {
            self.decoder: "decoder",
            self.causal_conv: "causal_conv",
            self.resample: "resample",
            self.upsample: "upsample",
            self.residual_block: "residual_block",
        }
        if self.residual_up_block is not None:
            roles[self.residual_up_block] = "residual_up_block"
        if self.chunked_attention and self.attention_block is not None:
            roles[self.attention_block] = "attention_block"
        return roles


WAN = WanVaeFamily(
    name="wan",
    vaes=(wan.AutoencoderKLWan,),
    decoder=wan.WanDecoder3d,
    encoder=wan.WanEncoder3d,
    causal_conv=wan.WanCausalConv3d,
    rms_norm=None,
    resample=wan.WanResample,
    upsample=wan.WanUpsample,
    residual_block=wan.WanResidualBlock,
    residual_up_block=wan.WanResidualUpBlock,
    dup_up=wan.DupUp3D,
    attention_block=wan.WanAttentionBlock,
)

_FAMILIES: list[WanVaeFamily] = [WAN]
_LAZY_LOADED = False


def _qwen_image() -> WanVaeFamily:
    q = importlib.import_module("diffusers.models.autoencoders.autoencoder_kl_qwenimage")
    return WanVaeFamily(
        name="qwen_image",
        vaes=(q.AutoencoderKLQwenImage,),
        decoder=q.QwenImageDecoder3d,
        encoder=q.QwenImageEncoder3d,
        causal_conv=q.QwenImageCausalConv3d,
        rms_norm=q.QwenImageRMS_norm,
        resample=q.QwenImageResample,
        upsample=q.QwenImageUpsample,
        residual_block=q.QwenImageResidualBlock,
        attention_block=q.QwenImageAttentionBlock,
        # QwenImageDecoder3d and QwenImageUpBlock have no ``first_chunk`` parameter.
        up_blocks_take_first_chunk=False,
    )


def _qwen_image_layered() -> WanVaeFamily:
    # vLLM-Omni's copy of the Qwen-Image VAE. Its RMS norm normalizes without the fp32 upcast, so it keeps its own
    # forward (``rms_norm=None``); every other role matches diffusers' Qwen-Image VAE.
    q = importlib.import_module("vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage")
    return WanVaeFamily(
        name="qwen_image_layered",
        vaes=(q.AutoencoderKLQwenImage,),
        decoder=q.QwenImageDecoder3d,
        encoder=q.QwenImageEncoder3d,
        causal_conv=q.QwenImageCausalConv3d,
        rms_norm=None,
        resample=q.QwenImageResample,
        upsample=q.QwenImageUpsample,
        residual_block=q.QwenImageResidualBlock,
        attention_block=q.QwenImageAttentionBlock,
        up_blocks_take_first_chunk=False,
    )


def _qwen_image_21() -> WanVaeFamily:
    q = importlib.import_module("vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21")
    return WanVaeFamily(
        name="qwen_image_21",
        vaes=(q.AutoencoderKLQwenImage21,),
        decoder=q.QwenImage21Decoder3d,
        encoder=q.QwenImage21Encoder3d,
        causal_conv=q.QwenImage21CausalConv3d,
        rms_norm=q.QwenImage21RMS_norm,
        resample=q.QwenImage21Resample,
        upsample=q.QwenImage21Upsample,
        residual_block=q.QwenImage21ResidualBlock,
        residual_up_block=q.QwenImage21ResidualUpBlock,
        dup_up=q.QwenImage21DupUp3D,
        attention_block=q.QwenImage21AttentionBlock,
        causal_conv_kind="conv2d_single_frame",
        chunked_attention=True,
    )


_LAZY_FAMILIES = (_qwen_image, _qwen_image_layered, _qwen_image_21)


def _load_lazy_families() -> None:
    # Imported on first use: the Qwen-Image-2.1 VAE lives under ``models/`` and must not be imported at package
    # import time. A family whose module is unavailable (older diffusers) is skipped.
    global _LAZY_LOADED
    if _LAZY_LOADED:
        return
    _LAZY_LOADED = True
    for build in _LAZY_FAMILIES:
        try:
            family = build()
        except (ImportError, AttributeError) as exc:
            logger.debug("Wan VAE fast path: family %s unavailable: %s", build.__name__, exc)
            continue
        if any(f.name == family.name for f in _FAMILIES):
            continue  # registered earlier (by the user): keep it
        register_family(family)


def register_family(family: WanVaeFamily) -> None:
    """Make ``family`` eligible for the fast path. Re-registering a family name replaces it."""
    _FAMILIES[:] = [f for f in _FAMILIES if f.name != family.name] + [family]
    _role_types.cache_clear()


def families() -> tuple[WanVaeFamily, ...]:
    _load_lazy_families()
    return tuple(_FAMILIES)


def find_family(vae: nn.Module) -> WanVaeFamily | None:
    """The registered family for ``vae``'s most specific VAE class (nearest in its MRO), or ``None``."""
    registered = families()
    for cls in type(vae).__mro__:
        for family in reversed(registered):  # the latest registration wins for the same class
            if cls in family.vaes:
                return family
    return None


def _collect(attr: str, *, kind: str | None = None) -> frozenset[type[nn.Module]]:
    out: set[type[nn.Module]] = set()
    for family in families():
        if kind is not None and family.causal_conv_kind != kind:
            continue
        cls = getattr(family, attr)
        if cls is not None:
            out.add(cls)
    return frozenset(out)


@cache
def _role_types(role: str) -> frozenset[type[nn.Module]]:
    if role == "causal_conv3d":
        return _collect("causal_conv", kind="conv3d")
    return _collect(role)


def is_role(module: object, role: str) -> bool:
    """Whether ``module``'s exact type plays ``role`` in a registered family.

    ``causal_conv3d`` matches only ``nn.Conv3d``-kind causal convs: call sites that inline ``F.conv3d`` or a
    temporally cached assembly must not take a single-frame ``Conv2d`` conv.
    """
    return type(module) in _role_types(role)


def rms_norm_types() -> frozenset[type[nn.Module]]:
    return _role_types("rms_norm")


def family_types(attr: str) -> Iterable[type[nn.Module]]:
    return _role_types(attr)


__all__ = [
    "CAUSAL_CONV_KINDS",
    "WAN",
    "WanVaeFamily",
    "families",
    "family_types",
    "find_family",
    "is_role",
    "register_family",
    "rms_norm_types",
]
