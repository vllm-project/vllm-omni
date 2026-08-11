# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-only precision configuration for structurally split Wan VAEs."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

PRECISION_TO_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _decode_with_precision(
    self: nn.Module,
    latents: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return type(self).decode(
        self,
        latents.to(dtype=self._vllm_decode_dtype),
        *args,
        **kwargs,
    )


def _encode_in_fp32(
    self: nn.Module,
    samples: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return type(self).encode(
        self,
        samples.to(dtype=torch.float32),
        *args,
        **kwargs,
    )


def configure_wan_decode_precision(vae: nn.Module, precision: str | None) -> nn.Module:
    """Keep Wan encode in FP32 while materializing only decode modules lower."""

    if precision is None:
        return vae

    from diffusers.models.autoencoders import AutoencoderKLWan

    if not isinstance(vae, AutoencoderKLWan):
        return vae
    try:
        decode_dtype = PRECISION_TO_DTYPE[precision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported vae_decode_precision={precision!r}; expected one of {sorted(PRECISION_TO_DTYPE)}"
        ) from exc

    encoder = getattr(vae, "encoder", None)
    quant_conv = getattr(vae, "quant_conv", None)
    decoder = getattr(vae, "decoder", None)
    post_quant_conv = getattr(vae, "post_quant_conv", None)
    if not all(isinstance(module, nn.Module) for module in (encoder, decoder, post_quant_conv)):
        logger.info("Wan VAE lacks separate encoder/decoder modules; skipping decode-only precision")
        return vae

    encoder.to(dtype=torch.float32)
    if isinstance(quant_conv, nn.Module):
        quant_conv.to(dtype=torch.float32)
    decoder.to(dtype=decode_dtype)
    post_quant_conv.to(dtype=decode_dtype)
    vae.encode = MethodType(_encode_in_fp32, vae)
    vae._vllm_decode_dtype = decode_dtype
    vae.decode = MethodType(_decode_with_precision, vae)
    logger.info(
        "Wan VAE configured with FP32 encode and %s decode",
        precision,
    )
    return vae


__all__ = ["PRECISION_TO_DTYPE", "configure_wan_decode_precision"]
